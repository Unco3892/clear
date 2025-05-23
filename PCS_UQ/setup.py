from setuptools import setup, find_packages

setup(
    name="pcs_uq",
    version="0.1.0",
    packages=['PCS', 'PCS.regression', 'PCS.classification', 'conformal_methods', 'metrics'],
    package_dir={
        'PCS': 'src/PCS',
        'conformal_methods': 'src/conformal_methods',
        'metrics': 'src/metrics',
    },
    install_requires=[
        # List your package dependencies here
        'requests>=2.25.1',
        'numpy>=1.21.5',
        'scikit-learn>=1.0.2',
        'pandas>=1.3.5',
        'matplotlib>=3.5.2',
        'seaborn>=0.11.2',
    ],
    author="Abhineet Agarwal, Michael Xiao, Rebecca Barter, Anthony Ozerov, Boyu Fan, Omer Ronen, Bin Yu",
    author_email="aa3797@berkeley.edu",
    description="A Python library for generating prediction intervals/sets via the PCS framework.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/abhineet-agarwal/pcs_uq",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
) 