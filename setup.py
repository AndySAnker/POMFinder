from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='POMFinder',
    version='1.0.1',
    author='Andy S. Anker',
    author_email='andy@chem.ku.dk',
    url='https://github.com/AndySAnker/POMFinder',
    description='Finds POM clusters from PDF data!',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=['POMFinder'],
    packages=['POMFinder'],
    package_dir={'POMFinder': 'src'},
    package_data={'POMFinder': ['Backend/*.model', 'Backend/*.h5', 'COD_ICSD_XYZs_POMs_unique99/*.xyz', 'Experimental_Data/*.gr']},

    entry_points = {'console_scripts': [
        'POMFinder=POMFinder.cli:main',
    ],
    },
    classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    ],
    include_package_data = True,
    zip_safe=False,

    install_requires=[
        'xgboost',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'h5py'
    ],
)

# requirements.txt for deployment on machines that you control.
# pip freeze to genereate requirements.txt file.
