# CIF-Finder
Welcome to CIF-Finder or __ciff__ for short!
This is a simple machine learning tool for structure characterisation of metal-oxides using total scattering Pair 
Distribution Function (PDF) analysis.  
Simply provide a PDF and the model will output best best structural models from it structure catalog which contain
approximately 11.000 metal oxides. 

1. [Install](#install)
2. [Usage](#usage)
3. [Authors](#authors)
4. [Cite](#cite)
5. [Acknowledgments](#acknowledgments)
6. [License](#license)
7. [Develop](#develop)

## Install
To install ciff you will need to have [Python](https://www.python.org/downloads/) or 
[Anaconda](https://www.anaconda.com/products/individual) installed. I recommend running ciff on Python version
3.7 or higher. If you have installed Anaconda you can create a new environment and activate it. 
```
conda create --name POMFinder_env python=3.7
conda activate POMFinder_env
```
Now you are ready to install what you actually come for! Currently __ciff__ is not avaible through PyPI or conda so the
package needs to be downloaded manually. The models are too big to be uploaded at GitHub (free version) so the
complete package can be downloaded [HERE](https://sid.erda.dk/sharelink/A82alE1lVb). Please note that there
might be several versions. Choose the desired version and download the .zip file within the folder. Exctract it and
navigate to the this package in the terminal you just created a new environment in. Run the following
command to install the __ciff__ package.  
```
pip install .
or
python setup.py install
```
It will take some time for it to install since some of the models are quite big and there is a lot of reference data.   
Or download wheel from releases and install it with pip.
```
pip install ciff-<version>-py3-none-any.whl
```
To verify that __ciff__ have been installed properly try calling the help argument.
```
ciff --help

>>> usage: ciff [-h] -d DATA [-m MODEL] [-n NTHREADS] [-s SHOW] [-p PEARSON]
>>>             [-o OUTPUT] [-f FILE_NAME] [-P PLOT]        
>>> 
>>> This is a package which takes a directory of PDF files 
>>> or a specific PDF file. It then determines the best structural 
>>> candidates based of a metal oxide catalog. Results can
>>> be compared with precomputed PDF through Pearson analysis. 
```  
This should output a list of possible arguments for running __ciff__ and indicates that it could find the package! 

## Usage
Now that __ciff__ is installed and ready to use, lets discuss the possible arguments. The arguments are described in 
greater detail at the end of this section.

| Arg | Description | Default |  
| --- | --- |  --- |  
|  | __Optional arguments__ | |  
| `-h` or `--help` | Prints help message. |    
| `-m` or `--model` | Choose what model to load. 0 large, 1 small and 2 both. __int__ | `-m 0`
| `-n` or `--nthreads` | Number of threads used by model. __int__ | `-n 1` 
| `-s` or `--show` | Number of best predictions printed. __int__ | `-s 5` 
| `-p` or `--pearson` | Calculate the Pearson correlation coefficient from pre-calculated PDFs of best suggested models. | `-p 5`
| `-o` or `--output` | Save a .csv with results. __bool__ | `-o True` 
| `-f` or `--file_name` | Name of the output file. __str__ | `-o ''` 
| `-P` or `--Plot` | Plot best prediction with loaded data. __int__ | `-P 5` 
|  | __Required argument__ | | 
| `-d` or `--data` | A directory of PDFs or a specific PDF file. __str__ | `-d 5` 

# Authors
__Emil T. S. Kjær__<sup>1</sup>  
__Andy S. Anker__<sup>1</sup>   
__Kirsten M. Ø. Jensen__<sup>1</sup>    
 
<sup>1</sup> Department of Chemistry and Nano-Science Center, University of Copenhagen, 2100 Copenhagen Ø, Denmark.   

Should there be any question, desired improvement or bugs please contact us on GitHub or 
through email: __etsk@chem.ku.dk__.

# Cite
If you use our code or our results, please consider citing our paper. Thanks in advance!
```
```

# Acknowledgments
Our code is developed based on the the following publication:
```
```

# License
This project is licensed under the Apache License Version 2.0, January 2004 - see the [LICENSE](LICENSE) file for details.

# Develop
Instal in developer mode.
```
$ pip install -e .[dev]
```
Build wheel from source distribution.
```
python setup.py bdist_wheel
```
