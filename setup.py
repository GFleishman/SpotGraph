import setuptools

setuptools.setup(
    name="SpotGraph",
    version="0.0.1",
    author="Greg M. Fleishman",
    author_email="greg.nli10me@gmail.com",
    description="Tools for dealing with smFISH spots and nuclei",
    url="https://github.com/GFleishman/SpotGraph",
    license="MIT",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'zarr',
        'numcodecs',
        'ClusterWrap',
        'dask',
        'dask[array]',
        'dask[delayed]',
        'dask[distributed]',
        'networkx',
    ]
)
