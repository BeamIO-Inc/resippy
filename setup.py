import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resippy",
    version="0.1.0",
    author="BeamIO, Inc.",
    author_email="info@beamio.net",
    description="REmote Sensing and Image Processing in PYthon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BeamIO-Inc/resippy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    license="BSD (3 clause)",
    install_requires=[
        'GDAL',
        'pyproj',
        'numpy<1.16.0',
        'scipy',
        'scikit-image',
        'shapely',
        'pint==0.9',
        'imageio',
    ]
)
