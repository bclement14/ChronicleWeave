#!/usr/bin/env python3
"""
Setup script for ChronicleWeave
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "ChronicleWeave - Audio Processing Pipeline"

# Read requirements if they exist
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "pathlib",
        "logging",
        "argparse",
    ]

setup(
    name="chronicleweave",
    version="1.0.0",
    description="Audio Processing Pipeline for Transcription and Script Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ChronicleWeave Team",
    author_email="contact@chronicleweave.com",
    url="https://github.com/chronicleweave/chronicleweave",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "chronicleweave": ["speaker_mapping.json", "modules/prompts/**/*"],
    },
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    entry_points={
        "console_scripts": [
            "chronicleweave=chronicleweave.cli:main",
        ],
    },
    keywords="audio transcription whisperx pipeline processing",
    project_urls={
        "Bug Reports": "https://github.com/chronicleweave/chronicleweave/issues",
        "Source": "https://github.com/chronicleweave/chronicleweave",
        "Documentation": "https://github.com/chronicleweave/chronicleweave#readme",
    },
) 