# Loan Default

This repository contains scripts and notebooks to:
- Analyze the relationship between different features and loan default status
- Build machine learning models to predict which companies will default on their loans
- Explain how different features impact model predictions

## Setup Instructions

The instructions here are required only if you wish to view/run the Jupyter Notebook on your local machine. Otherwise, you can just proceed to ***Viewing Instructions: To view the HTML version of the results***, which does not require any prior setup.

Run setup.sh to setup your conda python environment and install the necessary libraries. This set of instructions assumes that you are using a linux system with conda pre-installed.
```
chmod +x ./setup.sh
./setup.sh
conda activate loan-default
```

## Viewing Instructions

### To view the Jupyter Notebook version of the results:

- From the Home Page of the Jupyter Notebook, navigate to and open code_addison.ipynb.
- Run all the cells in the notebook from top to bottom (Need to execute this step to view the interactive visualizations).

### To view the HTML version of the results:

- Open code_addison.html in your browser.

### To view the helper scripts used in code_addison.ipynb:
- Navigate to and open utils/\_\_init\_\_.py

## Files

code_addison.ipynb
> - Jupyter Notebook containing the analysis results
> 
code_addison.html
> - HTML version of code_addison.ipynb
> 
utils/\_\_init\_\_.py
> - Contains helper classes and functions for analysis
> 
requirements.txt
> - Contains the list of dependencies required to run code_addison.ipynb and utils/\_\_init\_\_.py
> 
README.md
> - Contains the instructions for viewing the analysis results
>
data/train.csv
> - Provided train dataset for analysis and modelling
>
data/test.csv
> - Provided test dataset for model inference
>