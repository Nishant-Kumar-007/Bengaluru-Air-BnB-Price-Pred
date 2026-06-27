from setuptools import setup, find_packages

setup(
    name="BengaluruAirbnbPricePrediction",
    version="1.0.0",
    author="Nishant Kumar",
    description="End-to-end Airbnb price prediction for Bengaluru listings "
                "(DoorStep Analytics dataset)",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "catboost>=1.2",
        "flask>=3.0",
    ],
    python_requires=">=3.9",
)
