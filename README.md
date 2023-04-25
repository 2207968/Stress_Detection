# Stress_Detection

This repository contains stress detection  based on dryad dataset.  This dataset is a collection of biometric data of nurses during the COVID-19 outbreak. 

### Dataset Info
The Dryad Dataset is publicly available .Refer the following link to access data
https://datadryad.org/stash/dataset/doi:10.5061/dryad.5hqbzkh6f

### Environment Setup
1) Install Anaconda ( https://www.filehorse.com/download-anaconda-64/download/ )

### Steps for Data Loading and  preprocessing
1) Download the Dataset and copy it to 'Data/' Folder
2) Run 'unzip_script_1.py' for Perform Unzipping Operation
3) Run 'combine_data_script_2.py' for combining the different sensor values and store to csv files
4) Run 'merge_data_script_3.py' for Merging the csv files
5) Run 'label_data_script_4.py' for Labelling the merged data
