from setuptools import find_packages, setup

setup(
    name='roboverse',
    version='0.0.1',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Roboverse: Environments for Legged Robots',
    url="https://github.com/Winston-Gu/RoboVerse",  # Update this with your actual repository URL
    python_requires=">=3.8",
    install_requires=[
        'matplotlib',
        'numpy==1.23.5',
        'hydra-core>=1.2.0',
        'onnx',
        "tensorboard",
        "rich",
    ]
)