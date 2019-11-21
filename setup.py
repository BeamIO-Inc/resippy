import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resippy",
    version="0.3.0",
    author="BeamIO, Inc.",
    author_email="info@beamio.net",
    description="REmote Sensing and Image Processing in PYthon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BeamIO-Inc/resippy",
    include_package_data=True,
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
        'scipy',
        'scikit-image',
        'shapely',
        'exifread',
        'colorutils',
        'imageio',
        'numpy',
        'scikit-learn',
        'opencv-python',
        'pint',
        'seaborn',
        'utm'
    ],
)
