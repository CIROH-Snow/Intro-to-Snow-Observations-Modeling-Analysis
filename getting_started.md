# Introduction to Seasonal Snow Observations, Models, and Analysis

# Getting Started: 
Please fork this repo to your GitHub account.
Next, identify a folder location where you would like to work in a development environment.
Using the command prompt, change your working directory to this folder and git clone https://github.com/USERID/Intro-to-Snow-Observations-Modeling-Analysis

    git clone https://github.com/Intro-to-Snow-Observations-Modeling-Analysis


## Virtual Environment
It is a best practice to create a virtual environment when starting a new project, as a virtual environment essentially creates an isolated working copy of Python for a particular project. 
I.e., each environment can have its own dependencies or even its own Python versions.
Creating a Python virtual environment is useful if you need different versions of Python or packages for different projects.
Lastly, a virtual environment keeps things tidy, makes sure your main Python installation stays healthy and supports reproducible and open science.

## Creating Stable CONDA Environment on HPC platforms
Go to home directory
```
cd ~
```
Create an envs directory
```
mkdir envs
```
Create .condarc file and link it to a text file
```
touch .condarc

ln -s .condarc condarc.txt
```
Add the below lines to the condarc.txt file
```
# .condarc
envs_dirs:
 - ~/envs
```
Restart your server

### Creating your HydroLearnEnv Virtual Environment
Since we will be using Jupyter Notebooks for this exercise, we will use the Anaconda command prompt to create our virtual environment. 
We suggest using Mamba rather than conda for installs, conda may be used but will take longer.

For the learning activity in **Section 3** (i.e., 3.4), in the command line type: 

    mamba env create -f HydroLearnEnv.yml 

    conda activate HydroLearnEnv

    python -m ipykernel install --user --name=HydroLearnEnv

For the learning activity in **Section 4** (i.e., 4.5), in the command line type: 

    mamba env create -f HydroLearnNWMEnv.yml 

    conda activate HydroLearnNWMEnv

    python -m ipykernel install --user --name=HydroLearnNWMEnv
