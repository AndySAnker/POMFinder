[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/64e5fef7dd1a73847f5951b9)  |  [Paper]

# POMFinder
Welcome to POMFinder!
This is a simple machine learning tool for structure characterisation of polyoxometalate clusters using total scattering Pair 
Distribution Function (PDF) analysis.  
Simply provide a PDF and the model will output best best structural models from it structure catalog which contains 443 polyoxometalate clusters. 

1. [App](#app)
2. [Install](#install)
3. [Usage](#usage)
4. [Authors](#authors)
5. [Cite](#cite)
6. [License](#license)

## App
You can use POMFinder on a single PDF on the following HuggingFace App:
https://huggingface.co/spaces/AndySAnker/POMFinder

## Install
Install POMFinder with PyPi:
```
pip install POMFinder==1.0.0
```
More details about which Python versions are supported etc. can be found at https://pypi.org/project/POMFinder/1.0.0/.

To verify that __POMFinder__ has been installed properly, try calling the help argument.
```
POMFinder --help

>>> usage: POMFinder [-h] -d DATA [-n NYQUIST] [-i QMIN] [-a QMAX] [-m QDAMP] [-f FILE_NAME]       
>>> 
>>> This is a package which takes a directory of PDF files 
>>> or a specific PDF file. It then determines the best structural 
>>> candidates based on a polyoxometalate catalog. Results can
>>> be fitted to the PDF. 
```  
This should output a list of possible arguments for running __POMFinder__ and indicate that it could find the package! 

## Usage
Now that __POMFinder__ is installed and ready to use, let's discuss the possible arguments.

| Arg | Description | Default |  
| --- | --- |  --- |  
|  | __Required argument__ | | 
| `-h` or `--help` | Prints help message. |    
| `-n` or `--nyquist` | Is the data nyquist sampled. __bool__ | `-n True`
| `-i` or `--Qmin` | Qmin value of the experimental PDF. __float__ | `-i 0.7` 
| `-a` or `--Qmax` | Qmax value of the experimental PDF. __float__ | `-a 30` 
| `-m` or `--Qdamp` | Qdamp value of the experimental PDF. __float__ | `-m 0.04`
| `-f` or `--file_name` | Name of the output file. __str__ | `-o ''` 
| `-d` or `--data` | A directory of PDFs or a specific PDF file. __str__ | `-d 5` 

For example
```  
POMFinder --data "Experimental_Data/DanMAX_AlphaKeggin.gr" --nyquist "no" --Qmin 0.7 --Qmax 20 --Qdamp 0.02

>>> The 1st guess from the model is:  icsd_427457_1_0.9rscale.xyz
>>> The 2nd guess from the model is:  icsd_427379_0_0.9rscale.xyz
>>> The 3rd guess from the model is:  icsd_281447_0_1.0rscale.xyz
>>> The 4th guess from the model is:  icsd_423775_0_0.9rscale.xyz
>>> The 5th guess from the model is:  icsd_172542_0_1.1rscale.xyz

```  

**OBS:** As demonstrated in section E of the supplementary information, the Qmin, Qmax, and Qdamp values have only a minor impact on the prediction and can thus be omitted.

# Authors
__Andy S. Anker__<sup>1</sup>   
__Emil T. S. Kjær__<sup>1</sup>  
__Kirsten M. Ø. Jensen__<sup>1</sup>    
 
<sup>1</sup> Department of Chemistry and Nano-Science Center, University of Copenhagen, 2100 Copenhagen Ø, Denmark.   

Should there be any question, desired improvement or bugs please contact us on GitHub or 
through email: __andy@chem.ku.dk__.

# Cite
If you use our code or our results, please consider citing our paper. Thanks in advance!
```
https://chemrxiv.org/engage/chemrxiv/article-details/64e5fef7dd1a73847f5951b9
```

# License
This project is licensed under the Apache License Version 2.0, January 2004 - see the [LICENSE](LICENSE) file for details.
