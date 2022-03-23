from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='POMFinder',
    version='1.0.0',
    author='Andy S. Anker',
    author_email='andy@chem.ku.dk',
    url='https://github.com/AndyNano/POMFinder',
    description='Finds POM clusters from PDF data!',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=['POMFinder'],
    packages=['POMFinder'],
    package_dir={'POMFinder': 'POMFinder/Backend/'},
    package_data={'POMFinder': ['*.model', '*.h5', '*.xyz']},

    entry_points={'console_scripts': [
        'POMFinder=POMFinder.cli:main',
    ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
    include_package_data = True,
    zip_safe=False,

    install_requires=[
        'xgboost==1.5.2',
        'numpy==1.21.5',
        'matplotlib==3.5.1',
        'sklearn',
        'h5py'
    ],
)

# requirements.txt for deployment on machines that you control.
# pip freeze to genereate requirements.txt file.
