from pickle import LONG_BINGET
from re import L
from setuptools import setup

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


REPO_NAME= 'NLP_usecase_DVC'
AUTHOR_USER_NAME= 'ChanduKReddy99'
AUTHOR_EMAIL= 'chanduk.amical@gmail.com'
SRC_REPO= 'src'
PYTHON_REQUIRES= '>=3.6'
LIST_OF_REQUIREMENTS= [
    'tqdm',
    'dvc',
    'pandas',
    'numpy',
    'SciPy',
    'PyYAML',
    'scikit-learn',
    'lxml',
    'botcore'

]


setup(
    name=SRC_REPO,
    version= '0.0.2',
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description= 'A small NLP use case using DVC',
    long_description= long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/ChanduKReddy99/NLP_usecase_DVC',
    license='MIT',
    packages=[SRC_REPO],
    python_requires= PYTHON_REQUIRES,
    install_requires=LIST_OF_REQUIREMENTS,


)