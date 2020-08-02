from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import sys
import numpy as np
from typing import List, Union
from pathlib import Path
import os


def find_containing_folder(folders: List[str], filename: str) -> Union[str]:
    for f in folders:
        if Path(f).joinpath(filename).is_file():
            return f
    return None


library_dirs = list()
found_libde265 = False
for idx, argument in enumerate(sys.argv):
    if argument == "--libde265_path":
        library_dirs.append(sys.argv[idx+1])
        del sys.argv[idx+1]
        del sys.argv[idx]
        found_libde265 = True
        break
    elif argument.startswith("--libde265_path="):
        library_dirs.append(argument.split('=')[1])
        del sys.argv[idx]
        found_libde265 = True
        break
if not found_libde265:
    folder = find_containing_folder(os.environ.get('LD_LIBRARY_PATH').split(':'), "libde265.so")
    if folder is not None:
        library_dirs.append(folder)


extensions = [Extension("pyde265.image", ['pyde265/image.pyx'], libraries=['de265'], include_dirs=['include', np.get_include()],
                        library_dirs=library_dirs),
              Extension("pyde265.decoder", ['pyde265/decoder.pyx'], libraries=['de265'], include_dirs=['include', np.get_include()],
                        library_dirs=library_dirs)]

setup(
    name="PyDe265",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3, 'always_allow_keywords': True}),
    version="0.1.0",
    author="Jan",
    description="Python bindings for LibDe265",
    url="https://github.com/kloppjp/pyde265",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux"
    ],
    python_requires='>=3.6'
)
