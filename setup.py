from setuptools import setup, find_packages

setup(
    name='hym',
    version='v0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas'
    ],
)
