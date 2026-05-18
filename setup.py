from setuptools import setup, find_packages

setup(
    name="ca_aocp",
    version="0.1.0",
    description="Changepoint-Aware Adaptive Online Conformal Prediction",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=1.5",
        "matplotlib>=3.6",
    ],
    extras_require={
        "notebook": ["jupyter", "ipywidgets", "seaborn"],
    },
)
