import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Installs
setuptools.setup(
    name="d123",
    version="0.0.5",
    author="Daniel Dauner",
    author_email="daniel.dauner@gmail.com",
    description="TODO",
    url="https://github.com/autonomousvision/d123",
    python_requires=">=3.9",
    packages=["d123"],
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="apache-2.0",
    install_requires=requirements,
)
