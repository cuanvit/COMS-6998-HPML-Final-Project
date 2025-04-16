from setuptools import setup, find_packages

setup(
    name="own_gpt",
    version="0.1",
    packages=find_packages(),  # This will find the inner own_gpt/
    install_requires=["torch", "numpy"]
)