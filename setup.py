#!/usr/bin/env python3
"""
Setup script for multimodal-raptor package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multimodal-raptor",
    version="1.0.0",
    author="Taewoo Kim",
    author_email="twdaniel@gachon.ac.kr",
    description="High-performance multimodal transportation routing with Wave-Expansion and OSM Dijkstra RAPTOR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/multimodal-raptor",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "viz": ["folium>=0.12.0", "matplotlib>=3.4.0"],
        "dev": ["pytest>=6.0.0", "black>=21.0.0", "ruff>=0.1.0"],
    },
    entry_points={
        "console_scripts": [
            "mmraptor=mmraptor.pipeline.route:cli_main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)