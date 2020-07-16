from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import sys


library_dirs = list()
for idx, argument in enumerate(sys.argv):
    if argument == "--libde265_path":
        library_dirs.append(sys.argv[idx+1])
        del sys.argv[idx+1]
        del sys.argv[idx]
        break
    elif argument.startswith("--libde265_path="):
        library_dirs.append(argument.split('=')[1])
        del sys.argv[idx]
        break


extensions = [Extension("pyde265.image", ['pyde265/image.pyx'], libraries=['de265'], include_dirs=['include'],
                        library_dirs=library_dirs),
              Extension("pyde265.decoder", ['pyde265/decoder.pyx'], libraries=['de265'], include_dirs=['include'],
                        library_dirs=library_dirs)]

setup(
    name="PyDe265",
    ext_modules=cythonize(extensions, language_level=3),
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
