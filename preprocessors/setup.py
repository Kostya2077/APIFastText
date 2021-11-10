
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os
import Cython.Compiler.Options

import shutil


Cython.Compiler.Options.annotate = True


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'

class ExtensionNames:
    pass

class ExtensionNamesDir:
    pass





##################################################################
#################### MAIN INSTALL ################################
##################################################################

modules = []

for e in modules:
    e.cython_directives = {'language_level': "3"}

setup(
    name='all_modules',
    ext_modules=modules,
    cmdclass={'build_ext': build_ext},
    script_args=['build_ext'],
    options={'build_ext': {'inplace': True, 'force': True}},
)

############################ ПЕРЕНОС ФАЙЛОВ ПО ПАПКАМ ################################
names = []


files = [each for each in os.listdir(ROOT_DIR) if each.endswith('.so')]

for file in files:
    for name in names:
        for key in name:
            if key in file:
                shutil.move(ROOT_DIR + file, ROOT_DIR + name[key] + os.sep + file)