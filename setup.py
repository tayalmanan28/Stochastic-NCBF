from setuptools import setup

setup(
    name='Stochastic-NCBF',
    version='0.1',
    url='https://github.com/tayalmanan28/Stochastic-NCBF',
    description='Stochastic Neural Control Barrier Functions for safety verification of dynamical systems',
    author='Manan Tayal, Eric Zhang',
    author_email='manantayal@iisc.ac.in',
    packages=[
        'Inverted_pendulum',
        'unicycle_model',
        'deep_differential_network',
        'utils',
    ],
    classifiers=['Development Status :: 3 - Alpha'],
    install_requires=[
        'matplotlib',
        'numpy',
        'torch',
        'scipy',
    ],
    python_requires='>=3.8',
)
