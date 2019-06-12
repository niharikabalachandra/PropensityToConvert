# Propensity To Convert Models

# Abstract

The 80/20 rule has proven true for many businesses–only a small percentage of customers produce most of the revenue. Due to this, marketing teams are challenged to make appropriate investments in promotional strategies.

In this project, I analyzed a Google Merchandise Store (also known as GStore, where Google swag is sold) customer dataset to predict based on the web behavior of consumers on the Google Merchadise Online store, their likelihood or propensity to convert. 

Propensity models make true predictions about a customer’s future behavior. With propensity models you can truly anticipate a customer's future behavior.

Here we focus on building a combination of a Propensity to convert and a Propensity to buy model that can influence the kind of marketing campaigns we adopt and who we decide to target (predicted converters vs non-converters) leading to spend optimizations that eventually increase the ROI on digital marketing campaigns. 

The goal of building such propensity models is to attain more actionable operational changes and a better use of marketing budgets for those companies who choose to use data analysis on top of GA data.

# Introduction

Propensity models are what most people think of when they hear “predictive analytics” in the Marketing world. Propensity models make true predictions about a customer’s future behavior and help you truly anticipate a customers’ future behavior.

There are a number of types of Propensity models:

**Predicted lifetime value**

Algorithms can predict how much a customer will spend with you long before customers themselves realizes this. At the moment a customer makes their first purchase you may know a lot more than just their initial transaction record: you may have email and web engagement data for example, as well as demographic and geographic information. By comparing a customer to many others who came before him (or her) you can predict with a high degree of accuracy their future lifetime value. This information is extremely valuable as it allows you to make value based marketing decisions. For example, it makes sense to invest more in those acquisition channels and campaigns that produce customers with the highest predicted lifetime value.

**Predicted share of wallet**

With predicted share of wallet models you can estimate what percentage of a person’s category spend you currently have achieved. For example if a customer spends $100 with you on groceries, is this 10% or 90% of their grocery spending for a given year? Knowing this allows you to see where future revenue potential is within your existing customer base and to design campaigns to capture this revenue. 

**Propensity to convert**

The propensity to convert model can predict the likelihood for a customer to accept your offer. This model can be used for direct mail campaigns where the cost of marketing is high for example. In this case you only want to send the offers to customers with a high propensity to convert.

**Propensity to buy**

The propensity to buy model tells you which customers are ready to make their purchase: so you can find who to target. Moreover, once you know who is ready and who is not helps you provide the right aggression in your offer. Those that are likely to buy won't need high discounts while customers who are not likely to buy may need a more aggressive offer, thereby bringing you incremental revenue.

**Propensity to engage**

A propensity to engage model predicts how likely it is that a customer will click on your email links. Armed with this information you can decide not to send an email to a certain “low likelihood to click” segment.

**Propensity to unsubscribe**

A propensity to unsubscribe model predicts how likely it is that a customer will unsubscribe from your email list at any given point in time. Armed with this information you can optimize email frequency. For “high likelihood to unsubscribe” segments you should decrease send frequency, whereas for “low likelihood to unsubscribe” segments you can increase email send frequency.

**Propensity to churn**

The propensity to churn model tells you which active customers are at risk, so you know which high value, at risk customers to put on your watch list and reach out.

Often propensity models can be combined to make campaign decisions. For example, you may want to do an aggressive customer win back campaign for customers who have both a high likelihood to churn and a high predicted lifetime value.

Here we focus on building a combination of a Propensity to convert and a Propensity to buy model that can influence the kind of marketing campaigns we adopt and who we decide to target (predicted converters vs non-converters) leading to spend optimizations that eventually increase the ROI on digital marketing campaigns

# Data and Wrangling

Data used to build the Propensity to Convert model was obtained via the following Kaggle competition: https://www.kaggle.com/c/ga-customer-revenue-prediction

The train_v2.csv file contains the columns listed under **Data Fields**. Each row in the dataset is one visit to the Google Merchadise Online store and contains user transactions from August 1st 2016 to April 30th 2018

The dataset contains multiple columns which contain JSON blobs of varying depth. In one of those JSON columns, 'totals', the sub-column transactionRevenue contains the revenue information we use to build our targeting variable used for training and prediction:
* 1.0- Signifies a converter who generated revenue
* 0.0- Signifies a non-converter who did not generate revenue

## Data Fields

* **fullVisitorId -** A unique identifier for each user of the Google Merchandise Store.
* **channelGrouping -** The channel via which the user came to the Store.
* **date -** The date on which the user visited the Store.
* **device -** The specifications for the device used to access the Store.
* **geoNetwork -** This section contains information about the geography of the user.
* **socialEngagementType -** Engagement type, either "Socially Engaged" or "Not Socially Engaged".
* **totals -** This section contains aggregate values across the session.
* **trafficSource -** This section contains information about the Traffic Source from which the session originated.
* **visitId -** An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
* **visitNumber -** The session number for this user. If this is the first session, then this is set to 1.
* **visitStartTime -** The timestamp (expressed as POSIX time).
* **hits -** This row and nested fields are populated for any and all types of hits. Provides a record of all page visits.
* **customDimensions -** This section contains any user-level or session-level custom dimensions that are set for a session. This is a repeated field and has an entry for each dimension that is set.
* **totals -** This set of columns mostly includes high-level aggregate data.

# Workflow

1. [Pre-processing Pipeline to be Applied to the Data](#Preprocesing)
    * Hits and customDimensions do not have data that is relevant to predicting the target variable. Droping these columns:
    * normalizing the columns in json format and appending columns to original dataframe
    * Creating training label from the 'totals.transactionRevenue' column
    * Separating the converter and non-converter data to adress the class imbalance later in the processing pipeline
1. [Iterating Over the Data in Chunks Using nrows and skiprows](#BatchProcessing)
1. [Addressing Class Imbalance](#ClassImb)
    * Reading in all the data.
    * Set seed for random sampling.
    * Downsampling (without replacement) the non-converter data to arrive at a sample size equal to that of the converter data.
    * Split dataset into training and universal test data
    * Keep universal test data aside and ensure this data is noe present in future samples
    * Train model and save the model
    * Repeat steps 1-5 with different seeds to create many such training data draws.
    * Select model with heightest accuracy/ use weighted average of each model (based on accuracy) for prediction. Use same universal test dataset to determine accuracy of each model
1. [Converter Data (Label 1)](#GenLab)
1. [Split into Training and Test Data (80/20)](#Split)
1. [One-Hot Encoding of Features (Categorical Variables)](#OHE)
1. [Recursive Feature Elimination: Feature Importance](#RFE)
1. [Training the Model](#Training)
    * [Logistic Regression Model](#logreg)
    * [Random Forest Model](#RF)
1. [Exploratory Data Analysis to Create Scalable Data Pre-processing Pipeline and Feature Selection](#EDA)
    * Reading a subset of the data (.csv file train_v2.csv), about 20000 rows into memory
    * Converting fullVisitorId', 'visitId', 'visitStartTime' columns into string datatypes and flattening the 'totals', 'device', 'geoNetwork', 'trafficSource' columns which are in json format
    * Inspecting the contents of the columns containing json objects
    * Feature Selection
    * Dropping irrelevant features (low variance) and features with a large volume of missing data via inspection
    * Inspecting the contents of columns to determine which columns are worth keeping. Only columns with high variance is retained. Columns with missing data or low variance data are omitted.
    * Creating training label from the 'totals.transactionRevenue' column
    * We notice that only a small fraction of the visitors convert. This leads to a class imbalance that needs to be addressed before training our propensity to convert model.

# Environment

Jupyter Notebook which is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text is used here for data cleaning and transformation, statistical modeling and machine learning.

# Challenges

1. Dealing with a large dataset: Cloud vs local execution

1. Mixed dataset with regular text columns as well as json blobs

1. Dealing with the 80/20 rule that leads to class imbalance in training data

# Conclusion

Two Propensity to convert models were built using relativey low complexity machine learning algorithms, namely Logistic Regression and Random Forest. They had high accuracy scores of 95.56% and 95.73% on the test data.

The Random Forest model performed slightly better than the Logistic Regression model and is preferred due to faster training time as a result of paralization of the training process.

# Future work

Building out a regression model that predicts expected revenue for converters so as to better assign marketing budgets.

# References

https://blog.agilone.com/the-definitive-guide-to-predictive-analytics-models-for-marketing
