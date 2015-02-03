import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

source_path = os.path.abspath("pamutils")
sys.path.append(source_path)

import pamutils

packages = ["pamutils"]
requires = []

with open("README") as f:
    readme = f.read()
with open("LICENSE") as f:
    license = f.read()

setup(
    name="pamutils",
    version=pamutils.__version__,
    description="",  # TODO(SK): description missing
    long_description=readme,
    author="Sebastian Klatt",
    author_email="sebastian.klatt@rub.de",
    url="",  # TODO(SK): url missing
    packages=packages,
    package_data={"": ["LICENSE"]},
    package_dir={"": ""},
    include_package_data=True,
    install_requires=requires,
    license="GPL v2",
    zip_safe=False,
    classifiers=(
        # TODO(SK): classifiers missing
        # TODO(SK): full list is available at:
        # TODO(SK): https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Natural Language :: English',
    ),
)
