from setuptools import setup

setup(
    name='tracking-package',
    version='0.1.0',
    description='Python BBox Tracking Package',
    author='Alain Schöbi',
    author_email='alain.schoebi@gmx.ch',
    packages=['tracking'],
    install_requires=[
        'numpy',
        'matplotlib',
        'colorama',
        'pyyaml',
        'lapx',
        'cython-bbox',
    ],
)