# resippy
REmote Sensing and Image Processing in PYthon

Tools to perform image processing with a focus on overhead and spectral remote sensing applications.  Some features include:
Navigation from world coordinate to camera pixel coordinates and vice versa.
Spectral matched filters
Reading spectral libraries

Package will be available on PyPi and can be installed using pip.

A few notes on installation.  There are 2 dependencies that can be tricky.  Those dependencies are GDAL and PyProj.  An easy way to install these packages is through Anaconda.  If the user sets up a new Conda environment, and installs these packages first using:

conda install gdal
conda install pyproj

The rest of the installation should work.  Otherwise it is up to the user to install system dependencies to support installation of those packages through pip.
