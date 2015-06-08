try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Article Recommender System',
    'author': 'Tim van Cann',
    'url': '',
    'download_url': '',
    'author_email': 'timvancann@gmail.com',
    'version': '0.1',
    'install_requires': ['requirements.txt'],
    'packages': ['articles'],
    'scripts': [],
    'name': 'articles'
}

setup(**config)
