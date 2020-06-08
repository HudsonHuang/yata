from setuptools import setup, find_packages

import imp

version = imp.load_source('yata.version', 'yata/version.py')
description='Yet Another Tools for Audio deep learning'

with open('README.md') as file:
    long_description = file.read()

setup(
    name='libyata',
    version=version.version,
    description=description,
    author='HudsonHuang',
    author_email='790209714@qq.com',
    url='http://github.com/HudsonHuang/yata',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords='deep learning, audio processing, machine learning',
    license='MIT',
    install_requires=[
    ]
)
