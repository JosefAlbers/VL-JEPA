from setuptools import find_packages, setup

requirements = [
    "mlx==0.30.1; platform_system == 'Darwin'",
    "transformers",
    "datasets",
    "numpy",
    "Pillow",
]

extras_require = {
    'cuda': ['mlx[cuda]'],
    'cpu':  ['mlx[cpu]'],
    'no_mlx': [],
}

setup(
    name='vljepa',
    url='https://github.com/JosefAlbers/vljepa',
    packages=find_packages(),
    version='0.0.1a0',
    readme="README.md",
    author_email="albersj66@gmail.com",
    description="VL-JEPA (https://arxiv.org/abs/2512.10942)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="J Joe",
    license="Apache-2.0",
    # python_requires=">=3.12.8",
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "vlj=vljepa.main:cli",
        ],
    },
)
