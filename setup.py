from setuptools import setup

setup(
    name='tracking-package',
    version='0.0.1',
    description='Python BBox tracking package',
    author='Alain Schoebi',
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