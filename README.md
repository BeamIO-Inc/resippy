# ReSIPPy

#### Remote Sensing and Image Processing in Python

ReSIPPy is a set of tools to perform image processing with a focus on overhead and spectral remote sensing applications.
Some features include:
- Navigation from world coordinate to camera pixel coordinates
- Navigation from pixel coordinates to world coordinates
- Orthorectification
- Spectral matched filters
- Reading spectral libraries

### Installation

The installation of ReSIPPy requires two tricky dependencies:
- GDAL
- PyProj

#### Using conda

We prefer to use conda to manage our Python environments and dependencies.
To install Anaconda (or Miniconda), follow the instructions in the [Anaconda Docs](https://docs.anaconda.com/anaconda/install/).
Once you have a working conda installation, clone this repo and create an environment for working with ReSIPPy:

```bash
$ conda env create -f environment.yml
```

Then, activate the environment to work with ReSIPPy:

```bash
$ conda activate resippy
```

#### Using pip

ReSIPPy is also available on PyPi and can be installed using pip:

```bash
$ pip install resippy
```

Installation using pip may require the user to download and install additional system level dependencies.
