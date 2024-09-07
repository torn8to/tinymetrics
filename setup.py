from setuptools import setup, find_namespace_packages

setup(
    name='tinymetrics',
    description='A collection of loss functions',
    version='0.0.1',
    author='Nathan Rogers',
    license="MIT",
    packages=["tinymetrics","tinymetrics.losses"],
    install_requires=[
        "numpy", "tinygrad @ git+https://github.com/tinygrad/tinygrad.git"],
    python_requires='>=3.8',
    extras_require={"testing": ["torch", "torchmetrics"], })
