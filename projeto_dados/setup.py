from setuptools import find_packages, setup

setup(
    name="projeto_dados",
    packages=find_packages(exclude=["projeto_dados_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
