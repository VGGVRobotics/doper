from setuptools import setup, find_packages


setup(
    name="doper",
    version="0.0.1",
    description="End-to-end learning with differentiable dynamics",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "torch==1.4.0",
        "numpy==1.18.1",
        "jax==0.1.68",
        "jaxlib==0.1.47",
        "svgpathtools==1.3.3",
        "matplotlib",
        "pyyaml",
        "tripy",
    ],
)
