# Starbucks-Capstone

## Table of contents
* [Installation](#installation)
* [Project Motivation](#project-motivation)
* [File Description](#file-description)
* [Results](#results)
* [Acknowledgements](#authors-acknowledgements)
* [Setup](#setup)

## Installation
Project is created using:
* Python 3
* Pandas
* Numpy
* Scikit-learn
* Seaborn
* Matplotlib


## Project Motivation
The Starbucks Capstone details a test which was conducted on a sample of customers. The data captures customers' responses to different offers and the accompanying analyses aims to use the information to determine the effectiveness of these offers across the different customer segments.

My proposed solution leverages an understanding of customer purchase behaviour in framing an approach to segmenting customers within the sample and assessing the performance of each offer within the resulting customer segments. Specifically, the approach involves:

1. Reviewing customer transactions to identify those influenced by specific offers and those that were not. The resulting insights will be used to define individual customer behaviour/preferences (unstimulated spend, BOGO spend, Discount spend)

2. Segmenting the sample customer base using behaviour/preferences. Segments should contain customers of similiar purchase behaviour.

3. Reviewing value/count of offer-influenced transactions vs "uninfluenced" transactions at customer and segment levels. value/count of offer-influenced transactions represent incremental change due to offers while magnitude of change is established by comparing against relevant "uninfluenced" transaction measures. 
	
## File Description
1. Data Folder
* portfolio.json: Pipeline script to extract, transform and load data into a disaster_msg SQLite database
* profile.json: output of process_data.py
* transcript.json: input dataset containing messages sent by people during disasters.


2. Jupyter Notebook
* starbucksanalyse.ipynb


## Results
The following outcomes can be found within [this GitHub repository](https://github.com/ChidiOnum/Starbucks-Capstone.git)



## Authors, Acknowledgements
The data was provided by Starbucks and Udacity
