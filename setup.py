from setuptools import setup, find_packages

setup(
    name='pseudopruner',
    version='0.0',
    author='Zhouyang Xie',
    author_email='oceania.xie@gmail.com',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'torch>=1.6.0',
        'torchvision>=0.8.0'
    ],
)
