from setuptools import setup, find_packages


setup(
    name="doper",
    version="0.0.1",
    description="End-to-end locomotion learning with differentiable dynamics",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "taichi==0.6.5",
        "torch==1.4.0",
        "numpy==1.18.1",
        "svgpathtools==1.3.3",
        "matplotlib",
    ],
)
