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

Outcomes
* Customers' transactions were successfully labelled based on influencing offers - BOGO, Discount and Self.
* A customer-offer interaction matrix was generated with individual customer's preferences captured as unique combination of money spent across BOGO, Discount and Self. Thie was enriched with customer details from profile dataset to create a comprehensive reference dataset.
* PCA of key features and resulting K-means clustering generated five (5) customer segments with silhouette score of 0.7456 (cloer to 1, the better).
* Performance of offers was evaluated across different segments using transaction value and average transaction value. The choice of completion as an offer performance measure is insufficient as it excludes the cumulative impact of the small "unqualifying" transactions to the sales revenue of Starbucks (see category offer results below)

Description for Performance Measures
* cat_pop: number of customers in category
* recv: number of customers that received offer
* view: number of customers that viewed offer
* compltn: number of customers that completed offer after viewing
* notxns: number of transactions influenced by offer
* txnval: monetary value of transactions
* %compltn: percentage of customers who completed offer after viewing

Segment Offer Results

category results for bogo0505:
------------------------------
          cat_pop  recv  view  compltn  notxns    txnval  avgtxnval  %compltn
category                                                                     
0           14760  6563  6304     3146    7737  74856.13       9.68     47.94
1            1920   853   814      205     653   4909.77       7.52     24.03
2              57    15    15       11      13    320.44      24.65     73.33
3             208   100    93       66      91   1606.00      17.65     66.00
4              55    40    38       32      47  13164.03     280.09     80.00


category results for bogo0510:
------------------------------
          cat_pop  recv  view  compltn  notxns    txnval  avgtxnval  %compltn
category                                                                     
0           14760  6602  6345     2430    9122  80815.03       8.86     36.81
1            1920   839   805      139     716   4896.25       6.84     16.57
2              57    26    26       17      26    427.86      16.46     65.38
3             208    91    88       64      87   1613.62      18.55     70.33
4              55    35    34       26      42  10106.44     240.63     74.29


category results for bogo0705:
------------------------------
          cat_pop  recv  view  compltn  notxns    txnval  avgtxnval  %compltn
category                                                                     
0           14760  6707  3657     2156    4354  47525.56      10.92     32.15
1            1920   835   426      170     282   3088.96      10.95     20.36
2              57    24    12       10      16    311.14      19.45     41.67
3             208    92    62       50      71   1247.11      17.56     54.35
4              55    19    14       10      12   1783.61     148.63     52.63


category results for bogo0710:
------------------------------
          cat_pop  recv  view  compltn  notxns    txnval  avgtxnval  %compltn
category                                                                     
0           14760  6605  5787     2268   10374  85633.02       8.25     34.34
1            1920   886   788      175     935   6232.66       6.67     19.75
2              57    23    19       16      25    490.27      19.61     69.57
3             208   110    92       71     102   1707.85      16.74     64.55
4              55    34    30       23      34   9750.37     286.78     67.65


category results for disc0707:
------------------------------
          cat_pop  recv  view  compltn  notxns    txnval  avgtxnval  %compltn
category                                                                     
0           14760  6636  6373     3675    8634  84617.49       9.80     55.38
1            1920   864   825      242     821   6145.38       7.49     28.01
2              57    41    40       23      63   9809.09     155.70     56.10
3             208    85    80       66      91   1487.31      16.34     77.65
4              55    20    19       12      19    368.55      19.40     60.00


category results for disc0710:
------------------------------
          cat_pop  recv  view  compltn  notxns    txnval  avgtxnval  %compltn
category                                                                     
0           14760  6616  3588     2011    4299  49135.85      11.43     30.40
1            1920   876   431      139     310   3091.36       9.97     15.87
2              57    24    22       13      30   5306.13     176.87     54.17
3             208    93    61       48      61   1154.58      18.93     51.61
4              55    23    16       15      18    394.62      21.92     65.22


category results for disc1010:
------------------------------
          cat_pop  recv  view  compltn  notxns    txnval  avgtxnval  %compltn
category                                                                     
0           14760  6551  6321     3538   10328  97037.44       9.40     54.01
1            1920   871   833      204     936   6928.73       7.40     23.42
2              57    41    41       32      53  13831.52     260.97     78.05
3             208   112   111       97     129   2201.95      17.07     86.61
4              55    22    21       13      26    477.20      18.35     59.09


category results for disc1020:
------------------------------
          cat_pop  recv  view  compltn  notxns    txnval  avgtxnval  %compltn
category                                                                     
0           14760  6643  2283     1095    2925  37726.35      12.90     16.48
1            1920   876   314       92     231   2965.32      12.84     10.50
2              57    34    23       16      35   8575.51     245.01     47.06
3             208    90    32       18      45    750.99      16.69     20.00
4              55    25    11        6       8    143.29      17.91     24.00



6. Count and value of transactions completed under the influence of an offer provide a measure of the impact of the offer. These measures can be reviewed at individual and segment levels (Impact of offer - incremental transactions and revenue)




## Authors, Acknowledgements
The data was provided by Starbucks and Udacity
