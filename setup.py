#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages


def load_requirements(f):
    retval = [str(k.strip()) for k in open(f, "rt")]
    return [k for k in retval if k and k[0] not in ("#", "-")]


setup(
    name="leaf_cc",
    version="1.1.0",
    description="Project Reproducible Research with a Leaf Classification Problem",
    url="https://github.com/niedeado/M05_ProjectReproducibility",
    license="MIT",
    author="Fabio Mensi; Adi Niederberger",
    author_email="***",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    include_package_data=True,
    install_requires=load_requirements("requirements_pkg.txt"),
    entry_points={"console_scripts": ["leaf_cc-run_model = leaf_cc.main_script:main",
                                      "leaf_cc-run_test= leaf_cc.main_script:main_test"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
