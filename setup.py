from setuptools import setup, find_packages

setup(
    name="iconspy",
    version="0.1.0",
    description="ICON Sections in PYthon",
    author="Fraser William Goldsworth",
    author_email="frasergocean[at]gmail.com",
    url="https://github.com/fraserwg/iconspy",  # Replace with your repo URL
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add dependencies here, e.g., "Pillow>=8.0.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    license="BSD-3-Clause",
)