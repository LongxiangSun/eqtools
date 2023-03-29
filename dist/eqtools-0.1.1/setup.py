from distutils.core import setup
import setuptools
from distutils import util
from setuptools import find_packages, setup

pathcsiExtend = util.convert_path('src/csiExtend')

setup(name = 'eqtools',
    version = '0.1.1',
    author = 'kfh',
    author_email = 'kefenghe@whu.edu.cn',
    url = 'http://www.geologie.ens.fr/~kefenghe/eqtools/index.html',
    description = 'Earthquake related tools',
    package_dir = {
    'eqtools': 'src',
    'eqtools.csiExtend': pathcsiExtend,
    },
    packages=find_packages()
    )

# packages=['eqtools', 'eqtools.csiExtend']
# packages=setuptools.find_packages(where="eqtools")