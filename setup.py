from setuptools import setup, find_packages

setup(
    name="semantichar",
    version="0.2",
    install_requires=[
        "scikit-learn",
        "Pillow",
        "numpy",
        "ftfy",
        "regex",
        "einops",
        "iopath",
        "wandb",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
