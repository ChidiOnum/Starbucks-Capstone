# Starbucks-Capstone

## Table of contents
* [Installation](#installation)
* [Overview](#overview)
* [Approach](#approach)
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


## Overview
The Udacity Starbucks Capstone Project focuses on gaining key insights from an experiment conducted on a sample of Starbuck mobile app customers. The test was designed to generate data on customer behaviour in response to a set of offers and provide information required to identify new opportunities within new segments to drive sales. 

## Approach
My proposed solution leverages an understanding of customer purchase behaviour in framing an approach to segmenting customers within the sample and assessing the performance of each offer within the resulting customer segments. Specifically, the approach involves:

1. Reviewing customer transactions to identify those influenced by specific offers and those that were not. The resulting insights will be used to define individual customer behaviour/preferences (unstimulated spend, BOGO spend, Discount spend)

2. Segmenting the sample customer base using behaviour/preferences. Segments should contain customers of similiar purchase behaviour.

3. Reviewing transaction value/count of offer-influenced transactions vs "uninfluenced" transactions at customer and segment levels. The value/count of offer-influenced transactions represent incremental gains from customers responding to these offers while the extent of these gains is determined by using the relevant "uninfluenced" transaction as reference. 
	
## File Description
1. Data Folder

* profile.json: Rewards program users (17000 users x 5 fields)
Fields include:
1. gender: (categorical) M, F, O, or null
2. age: (numeric) missing value encoded as 118
3. id: (string/hash)
4. became_member_on: (date) format YYYYMMDD
5. income: (numeric)

* portfolio.json: Offers sent during 30-day test period (10 offers x 6 fields)
Fields include:
1. reward: (numeric) money awarded for the amount spent
2. channels: (list) web, email, mobile, social
3. difficulty: (numeric) money required to be spent to receive reward
4. duration: (numeric) time for offer to be open, in days
5. offer_type: (string) bogo, discount, informational
6. id: (string/hash)


* transcript.json: Event log (306648 events x 4 fields)
Fields:
1. person: (string/hash)
2. event: (string) offer received, offer viewed, transaction, offer completed
3. value: (dictionary) different values depending on event type
4. offer id: (string/hash) not associated with any "transaction"
5. amount: (numeric) money spent in "transaction"
6. reward: (numeric) money gained from "offer completed"
7. time: (numeric) hours after start of test

2. Jupyter Notebook
* starbucksanalyse.ipynb: Jupyter notebook with python codes to identify customer preferences/behaviour, identify opprtunity areas/segments with sample population and determine the performance of these offers within these opprtunity areas/segments.


## Results
The following outcomes can be found within [this GitHub repository](https://github.com/ChidiOnum/Starbucks-Capstone.git)

Background/Context:
* There are 2 states to customer behaviour: offer-induced ("excited") state and steady ("uninfluenced") state.
* Categorizing customer transactions by influence type - bogo, discount, self - and establishing the relative contribution of each category provides a quantitative definition of individual customer preferences/behaviour (Transaction Labelling).
* Transactions are said to be influenced by an offer when these transactions occur after a valid offer is received and viewed. In the event a transaction is completed before an offer is viewed, it is deemed to be uninfluenced (self). In addition, transactions completed outside the influence of any offer are deemed to be "uninfluenced" and labelled "self" (Transaction labelling)
* Informationals do not influence transactions because these do not have a motivator/reward (assumption)
* Customer behaviour can be used as a basis to segment customer base in addition to the use of demographic, psychograhic and other factors. There is a need to identify the most suitable factors (PCA and Clustering).
* Count and value of transactions completed under the influence of an offer provide a measure of the impact of the offer. These measures can be reviewed at individual and segment levels (Impact of offer - incremental transactions and revenue)

Outcomes
* Customers' transactions were successfully labelled based on influencing offers - BOGO, Discount and Self.
* A customer-offer interaction matrix was generated with individual customer's preferences captured as unique combination of money spent across BOGO, Discount and Self. Thie was enriched with customer details from profile dataset to create a comprehensive reference dataset.
* PCA of key features and resulting K-means clustering generated five (5) customer segments with silhouette score of 0.7456 (cloer to 1, the better).
* Performance of offers was evaluated across different segments using transaction value and average transaction value. The choice of completion as an offer performance measure is insufficient as it excludes the cumulative impact of the small "unqualifying" transactions to the sales revenue of Starbucks (see category offer results below)

see results in the [link]()

Segment Offer Results


## Authors, Acknowledgements
The data was provided by Starbucks and Udacity
