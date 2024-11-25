# Customer-Churn-Analysis
This is a python-based project which aims to predict whether a customer will churn based on their demographic details, service features, and account history.

<img width="599" alt="Image 2" src="https://github.com/user-attachments/assets/5d1e0e18-1f83-4396-859b-da8cc3996e2a">


# Introduction
Customer churn, also known as customer attrition, is when customers stop doing business with a company. It is a critical issue for businesses, especially those in highly competitive industries like telecommunications. The primary aim of this analysis is to identify patterns and factors contributing to customer churn at a fictional telecommunications company and use predictive models to forecast potential churners. 

# Problem Statement
The primary problem is to predict whether a customer will churn based on their demographic details, service features, and account history. Addressing this will help the company proactively identify at-risk customers and implement targeted retention strategies. What this means is that, we shall develop predictive models that help identify customers at risk of leaving. This insight can guide the company's customer retention strategies.

# Overview
In this project, where we shall use predictive models to forecast potential churners. Our predictions shall come under these four classification models: 
- Logistic Regression.
- Support Vector Classifier.
- Decision Tree Classifier.
- KNN Classifier.

# Dataset Overview
The dataset used in this analysis is titled "Telco Customer Churn" and includes data about customers and their service details. The dataset can be found here [Telco customer churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  The dataset can be summarized below: 

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents


# Importing Libraries 

We shall start by importing libraries which are necessary. 

<img width="203" alt="import libraries" src="https://github.com/user-attachments/assets/7303baad-1cc2-4481-b4ba-296416f39bf0">

# Loading dataset. 

So we moved on to create a dataframe and also checked the shape. 

<img width="865" alt="loading dataset" src="https://github.com/user-attachments/assets/bd8783a8-3c93-44f3-b712-fef779a570cc">

What this means is that, there are **7043** rows of data and **21** columns.


# Exploratory Data Analysis.

To take a brief look at our data, we used ```Df.head``` to view the first **5** top rows and then ```Df.tail``` to get the last **5** rows ```Df.shape``` as earlier used to check the number of rows and columns.
So the result of our examination of the table shows that all the columns are categorical column except: **SNRCITIZEN, TENURE, MONTHLYCHARGES, TOTALCHARGES**. That means we shall perform one-hot coding for all of them since they are numerical columns safe for the snrcitizen column since it is binary in nature.

I also noticed that most categorical columns have **4** or less unique values. E.g, in the “churn” column, we have just the “yes” and “no”.

We went on to check the size of the data using the ```Df.size``` code. which simply multiplied the number of rows by number of column to get **147903** ```Df.types```
to check the data type

<img width="214" alt="DATA TYPE" src="https://github.com/user-attachments/assets/866eacd5-0b0b-4cdb-b8f0-6222a01798a9">

Note that **totalchages** came as object instead of float. So we shall fix it.

So to check again the columns of the dataset, we used ```df.columns``` as can be seen here

<img width="437" alt="DF COLUMNS" src="https://github.com/user-attachments/assets/0b5abde1-a6b3-442b-aee3-73ebd8305a08">

then ```df.info``` to get the information about each column 

<img width="314" alt="DF INFO MAIN" src="https://github.com/user-attachments/assets/766d895f-30ba-45ff-b4fa-2cdf1ca157c2">

From the image, we can see that there are no null values (no-null) however, we can confirm this with ```df.isnull().sum()``` code

<img width="152" alt="null value" src="https://github.com/user-attachments/assets/5911673c-57f3-47aa-9900-f2aea729858b">

So from the image above, we can see that all indicate 0 null values.

I also checked to see if there are duplicate values and saw that there are no duplicate values for the dataframe.
So having identified some isuues in our dataset, we can move on to cleaning our data.
 
# BASIC DATA CLEANING
As observed, the totalcharges was object datatype, so we changed to float. Moving further, I tried to find the correlation between the numerical features. 

<img width="379" alt="numerical correlation" src="https://github.com/user-attachments/assets/c214e975-1fe4-4f4e-8ff2-0470f8ac8be0">

# FEATURE DISTRIBUTION

To check  for outliers and compare feature distributions with target variable, we went on to plot distributions for numerical and categorical features.
Numerical Features distribution.

To get the summary statistics of the three numerical features (**tenure, MonthlyCharges, and TotalCharges**) in the dataset, we used the code : ```df[numerical_features].describe()```
So the distribution of the numerical features can be seen in the image below:

<img width="295" alt="Numerical Feature distribution" src="https://github.com/user-attachments/assets/4b5502bf-f84a-4102-9dbf-6c5d5bc35de6">

From the image, we can understand the following:
- Count: Here we try to see how many customers have data in this column.
  - Tenure and MonthlyCharges have data for 7,043 customers.
  - TotalCharges has data for 7,032 customers — so 11 customers are missing this information.
    
- mean (average):
  - On average, customers stay for about 32 months (tenure), pay $64.76 per month (MonthlyCharges), and have paid a total of $2,283.30 so far (TotalCharges).
    
- Std (standard deviation): This tells us how spread out the data is.
  - For tenure, we can say on average, customers’ tenure varies by about 24.55 months from the mean of 32 months. That means most customers are likely to have stayed for 32 ± 24.55 months.
  - MonthlyCharges varies by $30. Some people pay much less, and some pay much more than the average.
  - For Totalcharges, it means that the total amount customers have paid varies by about $2,266.77 from the average of $2,283.30. This means that most customers have likely paid between $16.53 and $4,550.07

- Min (minimum) and max (maximum):
  - The shortest time a customer has stayed is 0 months, and the longest is 72 months.
  - The lowest monthly charge is $18.25, and the highest is $118.75.
  - The smallest total charge paid is $18.80, and the largest is $8,684.80.
    
- 25%, 50% (median), 75%: These tell you how the data is divided into parts:
  - 25% of customers have been with the company for 9 months or less.
  - Half (50%) of the customers have been with the company for 29 months or less.
  - 75% of customers have stayed for 55 months or less.


Now to consider the distribution of numerical features in relation to the target variable
<img width="883" alt="numerical feature in relation to churn" src="https://github.com/user-attachments/assets/3b0473ad-730b-48b7-8edd-773c52053dcc">

For easy understanding, blue color is “not churn”, while red color is “churn”

<img width="883" alt="numerical feature in relation to churn" src="https://github.com/user-attachments/assets/b33df0d1-d861-449c-91ec-5b5a26bfbf4c">

