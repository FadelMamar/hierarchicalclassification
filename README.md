# hierarchicalclassification
The current directory provides the code necessary to reproduce the results from my Master Thesis report.
For any addtional question, send me an email: fadel.seydou@gmail.com.

## Installing dependencies
- Open a terminal on Linux or a _Anaconda_ command prompt on Windows
- Clone this repository
- Move to the repository with ```cd ./hierarchicalclassification```
- Create a virtual environment
    - ```python -m venv .venv```
- Activate a virtual environment 
    - For Linux: ```source ../.venv/bin/activate```
    - For Windows: ```.venv\bin\activate.bat```
- Install dependencies
    - ```pip install -r requirements.txt```

## Project workflow
- Open the jupyter notebook: ```./src/data_exploration.ipynb```
- It will explain the general workflow and hypothesis behind the data prepration.

## Data preparation
- Download the following files and move them to ```./data```:
    - [https://1drv.ms/u/s!ArkwIkcloZytkdd_Dae2862tU5KSNA?e=E55c5w] brasil_coverage_2018_labelDist_2.csv
    - [https://1drv.ms/i/s!ArkwIkcloZytjcF6JAnR2pBZgJ41vw?e=QCt5PQ] brasil_coverage_2018.tif
- Move to directory ```./src```
- Run ```python datapreparation.py``` to create needed files for downloading data
- Initialize to Google earth engine API: [https://developers.google.com/earth-engine/guides/python_install#authentication] here
- Run ```bash download_data.sh```

## Training
- In the terminal, move to ```./src```
- Create a free account on [https://wandb.ai/]weight&biases to log training metrics
- Initialize weight&biases in this working directory as follows: ```wandb login``` and provide API key.
- Open ```train.sh``` and update the parameters. All parameters are explained in ```args.py```
- Run ```bash train.sh```