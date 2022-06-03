from setuptools import find_packages, setup

VERSION = "0.1"
DESCRIPTION = "Package Hamiltonians for different states of TlF."

setup(
    name="TlF_hamiltonians",
    version=VERSION,
    author="Oskari Timgren",
    description=DESCRIPTION,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.5",
        "sympy>=1.9",
        "joblib",
    ],
)
