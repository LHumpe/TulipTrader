from setuptools import setup, find_namespace_packages

setup(
    name='TulipTrader',
    version='0.1',
    description='',
    author='Lukas Humpe',
    author_email='l.humpe@hotmail.de',
    packages=find_namespace_packages(include=['tuliptrader']),
    install_requires=['pandas==1.2.3',
                      'stockstats==0.3.2',
                      'flask==1.1.2',
                      'requests==2.25.1',
                      'numpy==1.20.2',
                      'click==7.1.2',
                      'stable-baselines3==1.0',
                      'gym==0.18.0'],
    entry_points={
        'console_scripts': ['crypto=tuliptrader.crypto.cli:cli']
    }
)
