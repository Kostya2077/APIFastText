
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
    TEXT_NORMALIZER = 'text_normalizer'
    TF_IDF = 'tf_idf'

class ExtensionNamesDir:
    TEXT_NORMALIZER = {ExtensionNames.TEXT_NORMALIZER: 'text_normalizer'}
    TF_IDF = {ExtensionNames.TF_IDF: 'tf_idf'}


text_normalizer = Extension(ExtensionNames.TEXT_NORMALIZER,
                sources=[ROOT_DIR + 'text_normalizer/text_normalizer.pyx'],
                extra_compile_args=["-std=c++11", "-Ofast",  "-ftree-vectorize", "-msse2"],
                extra_link_args=["-std=c++11", "-Ofast",  "-ftree-vectorize", "-msse2"],
                include_dirs=[ROOT_DIR, '.'],
                language='c++')

tf_idf = Extension(ExtensionNames.TF_IDF,
                sources=[ROOT_DIR + 'tf_idf/tf_idf.pyx'],
                extra_compile_args=["-std=c++11", "-Ofast",  "-ftree-vectorize", "-msse2"],
                extra_link_args=["-std=c++11", "-Ofast",  "-ftree-vectorize", "-msse2"],
                include_dirs=[ROOT_DIR, '.'],
                language='c++')




##################################################################
#################### MAIN INSTALL ################################
##################################################################

modules = [
    text_normalizer,
    tf_idf
]

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
names = [ExtensionNamesDir.TF_IDF, ExtensionNamesDir.TEXT_NORMALIZER,]

files = [each for each in os.listdir(ROOT_DIR) if each.endswith('.so')]

for file in files:
    for name in names:
        for key in name:
            if key in file:
                shutil.move(ROOT_DIR + file, ROOT_DIR + name[key] + os.sep + file)