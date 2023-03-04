"""Setup script."""

import os
import pathlib

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

BUILD_WITH_CUSTOM_OPS = (
    "BUILD_WITH_CUSTOM_OPS" in os.environ
    and os.environ["BUILD_WITH_CUSTOM_OPS"] == "true"
)

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return BUILD_WITH_CUSTOM_OPS

    def is_pure(self):
        return not BUILD_WITH_CUSTOM_OPS


setup(
    name="stable-diffusion",
    description="tensorflow project with stable diffusion",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/tensorflow-project/stable-diffusion",
    author="tensorflow-project",
    author_email="aschirrmeist@uni-osnabrueck.de",
    license="Apache License 2.0",
    install_requires=["packaging", "absl-py", "regex", "tensorflow-datasets"],
    python_requires=">=3.7",
    extras_require={
        "tests": [
            "flake8",
            "isort",
            "black[jupyter]",
            "pytest",
            "pycocotools",
            "tensorflow",
        ],
        "examples": ["tensorflow_datasets", "matplotlib"],
    },
    distclass=BinaryDistribution,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    packages=find_packages(exclude=("*_test.py",)),
    include_package_data=True,
