# Copyright 2025 J Joe

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import json
import time
import math
from types import SimpleNamespace
import requests

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
from mlx.utils import tree_unflatten
from PIL import Image

CFG_L = dict(rms_norm_eps = 1e-6, rope_base = 10000.0, attn_bias = False)
CFG_V = dict(image_size = 224, num_channels = 3, layer_norm_eps = 1e-6, attn_bias = True)
URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

class DoRALinear(nn.Module):
    @staticmethod
    def from_linear(linear, r, alpha, scale, dropout):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = DoRALinear(input_dims=input_dims, output_dims=output_dims, r=r, alpha=alpha, scale=scale, dropout=dropout)
        lora_lin.linear = linear
        return lora_lin

    def __init__(self, input_dims, output_dims, r, alpha, scale, dropout, bias=False):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale * (alpha / r)
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-init_scale, high=init_scale, shape=(input_dims, r))
        self.lora_b = mx.zeros(shape=(r, output_dims))
        self.m = mx.linalg.norm(self._dequantized_weight().astype(mx.float32), axis=1)

    def _dequantized_weight(self):
        weight = self.linear.weight
        if isinstance(self.linear, nn.QuantizedLinear):
            weight = mx.dequantize(weight, self.linear.scales, self.linear.biases, self.linear.group_size, self.linear.bits)
        return weight

    def __call__(self, x):
        bias = self.linear.bias if "bias" in self.linear else 0
        y = self.linear(x)
        y = y - bias
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        z = y + (self.scale * z)
        w = self._dequantized_weight()
        adapted = w + (self.scale * self.lora_b.T) @ self.lora_a.T
        denom = mx.linalg.norm(adapted, axis=1) + 1e-6
        z = (self.m / denom) * z
        z = z + bias
        return z

def to_dora(layers, targets=None, rank=8, scale=0.1):
    _targets = ['o_proj', 'down_proj'] if targets is None else targets
    for l in layers:
        loralized = [(k, DoRALinear.from_linear(m, r=rank, alpha=rank, scale=scale, dropout=0.0)) for k, m in l.named_modules() if any(k.endswith(_t) for _t in _targets)]
        l.update_modules(tree_unflatten(loralized))

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dims = config.hidden_size
        bias = config.attn_bias
        self.n_heads = n_heads = config.num_attention_heads
        head_dim = dims // n_heads
        self.n_kv_heads = n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dims, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dims, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dims, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dims, bias=bias)
        if getattr(config, 'rope_base', False):
            self.rope = nn.RoPE(head_dim, base = config.rope_base)
        else:
            self.rope = lambda x, *args, **kwargs: x

    def __call__(self, x, mask=None, cache = None):
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output), (keys, values)

class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.projection_dim, bias=True)

    def __call__(self, x):
        return self.linear(x)

class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.num_patches = (config.image_size // config.patch_size) ** 2 
        self.position_embedding = nn.Embedding(self.num_patches, config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.flatten(self.patch_embedding(x), start_axis=1, end_axis=2) + self.position_embedding(mx.arange(self.num_patches)[None, :])

class GELU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu_approx(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = GELU(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        r, _ = self.self_attn(self.layer_norm1(x))
        h = x + r
        r = self.mlp(self.layer_norm2(h))
        return h + r

class VisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x):
        x = self.embeddings(x)
        for l in self.layers:
            x = l(x)
        x = self.post_layernorm(x) 
        return x

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = mx.ones((config.hidden_size,))
        self.eps = config.rms_norm_eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)

class GeGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = GeGLU(config)
        self.input_layernorm = RMSNorm(config)
        self.post_attention_layernorm = RMSNorm(config)

    def __call__(self, x, mask = None, cache = None):
        r, cache = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out, cache

class LanguageModel(nn.Module):
    def __init__(self, config, num_layers_override=None):
        super().__init__()
        self.scale = config.hidden_size**0.5
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config=config) for _ in range(config.num_hidden_layers if num_layers_override is None else num_layers_override)]
        self.norm = RMSNorm(config)

    def __call__(self, input_ids, inputs_embeds=None, attention_mask_4d=None, cache=None, output_hidden_states=False):
        cache = [None] * len(self.layers) if cache is None else cache
        h = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        h = h * self.scale
        for e, layer in enumerate(self.layers):
            h, cache[e] = layer(h, attention_mask_4d, cache[e])
        if output_hidden_states:
            return self.norm(h)
        return self.embed_tokens.as_linear(self.norm(h)), cache

class PG(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        model_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors", "*.json"], token=os.getenv('HF_TOKEN'))
        config = _get_cfg(f"{model_path}/config.json")
        config.vision_config = SimpleNamespace(**(CFG_V|config.vision_config))
        config.text_config = SimpleNamespace(**(CFG_L|config.text_config))
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = Projector(config.vision_config)
        _get_wt(model_path, config, model=self)

class X_Encoder(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.vision_tower = VisionModel(config)
        self.multi_modal_projector = Projector(config) 

    def __call__(self, x_v):
        x_v = self.vision_tower(x_v) 
        x_v = self.multi_modal_projector(x_v)
        return x_v

class Y_Encoder(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.language_model = LanguageModel(config, num_layers_override=num_layers)
    def __call__(self, y):
        y = self.language_model(y, output_hidden_states=True)
        return mx.mean(y, axis=1)

class Predictor(nn.Module):
    def __init__(self, config, num_layers):
        super().__init__()
        self.language_model = LanguageModel(config, num_layers_override=num_layers)
    def __call__(self, s_v, query_ids):
        s_v = s_v / self.language_model.scale
        x_q = self.language_model.embed_tokens(query_ids)
        x = mx.concatenate([s_v, x_q], axis=1)
        x = self.language_model(input_ids=None, inputs_embeds=x, output_hidden_states=True)
        return mx.mean(x, axis=1)

class Y_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_model = LanguageModel(config)
    def __call__(self, s_y):
        logits, cache = self.language_model(input_ids=None, inputs_embeds=s_y)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        list_tokens = token.tolist()
        for _ in range(100):
            logits, cache = self.language_model(token[None], None, None, cache)
            token = mx.argmax(logits[:, -1, :], axis=-1)
            list_tokens += token.tolist()
            # if list_tokens[-1] == processor.tokenizer.eos_token_id:
            #     break
        return list_tokens

class VLJEPA(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        model_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors", "*.json"], token=os.getenv('HF_TOKEN'))
        config = _get_cfg(f"{model_path}/config.json")
        config.vision_config = SimpleNamespace(**(CFG_V|config.vision_config))
        config.text_config = SimpleNamespace(**(CFG_L|config.text_config))
        self.config=config
        self.x_encoder = X_Encoder(config.vision_config, num_layers=4)
        self.y_encoder = Y_Encoder(config.text_config, num_layers=4)
        self.predictor = Predictor(config.text_config, num_layers=4)
        self.y_decoder = Y_Decoder(config.text_config)
        _get_wt(model_path, config, model=self.x_encoder)
        _get_wt(model_path, config, model=self.y_encoder)
        _get_wt(model_path, config, model=self.predictor)
        _get_wt(model_path, config, model=self.y_decoder)

    def __call__(self, pixel_values, query_ids):
        s_v = self.x_encoder(pixel_values)
        return self.predictor(s_v, query_ids)

def _get_cfg(json_path, **kwargs):
    try:
        with open(json_path, "r") as f:
            cfg = SimpleNamespace(**(json.load(f)|kwargs))
        return cfg
    except:
        return False

def _get_wt(model_path, model_cfg, model=None):
    if getattr(model_cfg, 'sanitized', False):
        wt = [(k, v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    else: 
        wt = [(k.replace('vision_tower.vision_model.', 'vision_tower.').replace('language_model.model.', 'language_model.').replace('encoder.layers.', 'layers.').replace('self_attn.out_proj.','self_attn.o_proj.'), v.transpose(0, 2, 3, 1) if "patch_embedding.weight" in k else v) for wf in glob.glob(f"{model_path}/*.safetensors") for k, v in mx.load(wf).items()]
    if model is None:
        return wt
    ok = [i[0] for i in model.named_modules()]
    wt = [i for i in wt if any(i[0].startswith(_k) for _k in ok)]
    model.load_weights(wt, strict=False)

def infonce_loss(predicted_embeddings, target_embeddings, temperature=0.07):
    pred_norm = predicted_embeddings / (mx.linalg.norm(predicted_embeddings, axis=-1, keepdims=True) + 1e-6)
    target_norm = target_embeddings / (mx.linalg.norm(target_embeddings, axis=-1, keepdims=True) + 1e-6)
    logits = (pred_norm @ target_norm.T) / temperature
    B = logits.shape[0]
    labels = mx.arange(B)
    return mx.mean(nn.losses.cross_entropy(logits, labels))

def train_vljepa(model, dataset_iterator, n_epoch=2, lr=1e-4):
    model.freeze()
    to_dora(model.predictor.language_model.layers)
    def loss_fn(model, images, query_ids, target_ids):
        target_emb = model.y_encoder(target_ids)
        pred_emb = model(images, query_ids)
        return infonce_loss(pred_emb, target_emb)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=lr)
    for epoch in range(n_epoch):
        total_loss = 0
        steps = 0
        tic = time.perf_counter()
        for batch in dataset_iterator():
            images, queries, answers = batch
            loss, grads = loss_and_grad(model, images, queries, answers)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_loss += loss.item()
            steps += 1
            if steps >= 10: break 
        print(f"Epoch {epoch+1}: Loss {total_loss/steps:.4f} ({time.perf_counter() - tic:.2f}s)")

def get_overfit_batch(processor, batch_size=2):
    car_image = Image.open(requests.get(URL, stream=True).raw)
    car_text = "A car"
    noise_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    noise_text = "Noise"
    query_text = "Caption: What is this?"
    while True:
        imgs = [car_image, noise_image]
        pixel_values = processor.image_processor(imgs, return_tensors="np").pixel_values
        q_tokens = processor.tokenizer([query_text, query_text], return_tensors="np", padding=True).input_ids
        a_tokens = processor.tokenizer([car_text, noise_text], return_tensors="np", padding=True).input_ids
        img_batch = mx.array(pixel_values).transpose(0, 2, 3, 1)
        q_batch = mx.array(q_tokens) 
        a_batch = mx.array(a_tokens) 
        yield img_batch, q_batch, a_batch

def test_sanity(model_id="google/paligemma-3b-mix-224"):
    model = PG(model_id)
    mx.eval(model.parameters())
    processor = AutoProcessor.from_pretrained(model_id)
    raw_image = Image.open(requests.get(URL, stream=True).raw)
    print(f'{raw_image=}')
    pixel_values = processor.image_processor(raw_image, return_tensors="np").pixel_values
    pixel_values = mx.array(pixel_values).transpose(0, 2, 3, 1)
    vis_features = model.vision_tower(pixel_values)
    vis_features = model.multi_modal_projector(vis_features)
    vis_features = vis_features / model.language_model.scale
    prompt = "What is this?"
    input_ids = processor.tokenizer(prompt, return_tensors="np").input_ids
    input_ids = mx.array(input_ids)
    text_embeds = model.language_model.embed_tokens(input_ids)
    curr_input = mx.concatenate([vis_features, text_embeds], axis=1)
    print(f"Prompt: {prompt}")
    print("Output: ", end="", flush=True)
    logits, cache = model.language_model(
        input_ids=None,
        inputs_embeds=curr_input,
        cache=None
    )
    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    print(processor.decode([next_token.item()]), end="", flush=True)
    for i in range(20):
        curr_input = mx.expand_dims(model.language_model.embed_tokens(next_token), axis=1)
        logits, cache = model.language_model(
            input_ids=None, 
            inputs_embeds=curr_input, 
            cache=cache
        )
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = next_token.item()
        if token_id == processor.tokenizer.eos_token_id:
            break
        print(processor.decode([token_id]), end="", flush=True)

def test_retrieval(model, processor):
    raw_image = Image.open(requests.get(URL, stream=True).raw)
    pixel_values = processor.image_processor(raw_image, return_tensors="np").pixel_values
    img = mx.array(pixel_values).transpose(0, 2, 3, 1)
    q_ids = mx.array(processor.tokenizer("Caption: What is this?", return_tensors="np").input_ids)
    a_car_ids = mx.array(processor.tokenizer("A car", return_tensors="np").input_ids)
    a_noise_ids = mx.array(processor.tokenizer("Noise", return_tensors="np").input_ids)
    pred_emb = model(img, q_ids) # [1, Dim]
    target_car = model.y_encoder(a_car_ids)
    target_noise = model.y_encoder(a_noise_ids)
    pred_norm = pred_emb / mx.linalg.norm(pred_emb, axis=-1, keepdims=True)
    car_norm = target_car / mx.linalg.norm(target_car, axis=-1, keepdims=True)
    noise_norm = target_noise / mx.linalg.norm(target_noise, axis=-1, keepdims=True)
    score_car = (pred_norm @ car_norm.T).item()
    score_noise = (pred_norm @ noise_norm.T).item()
    print(f"Similarity to 'A car': {score_car:.4f}")
    print(f"Similarity to 'Noise': {score_noise:.4f}")

def main():
    model_id = "google/paligemma-3b-mix-224"
    model = VLJEPA(model_id)
    mx.eval(model.parameters())
    processor = AutoProcessor.from_pretrained(model_id)
    train_vljepa(model, lambda: get_overfit_batch(processor), n_epoch=1, lr=1e-4)
    test_retrieval(model, processor)

def cli():
    test_sanity()

if __name__ == '__main__':
    main()
