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

To take a brief look at our data, we used ```Df.head``` to view the first **5** top rows.

<img width="823" alt="DF HEAD 1" src="https://github.com/user-attachments/assets/33de0e94-bca3-4659-af6b-efe1ac742703">

<img width="640" alt="DF HEAD 2" src="https://github.com/user-attachments/assets/7e6b48bc-527d-496f-8cb7-6d56e6776fe1">

Then ```Df.tail``` to get the last **5** rows ```Df.shape``` as earlier used to check the number of rows and columns.
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
 
# Basic data cleaning
As observed, the totalcharges was object datatype, so we changed to float. Moving further, I tried to find the correlation between the numerical features. 

<img width="379" alt="numerical correlation" src="https://github.com/user-attachments/assets/c214e975-1fe4-4f4e-8ff2-0470f8ac8be0">

# Feature Distribution

To check  for outliers and compare feature distributions with target variable, we went on to plot distributions for numerical and categorical features.

### Numerical Features distribution.

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

What the image above tries to tell us is that:


1) For the Tenure (How long customers stayed) visual we can see that: 

<img width="290" alt="Tenure numerical Feature" src="https://github.com/user-attachments/assets/5b60ad37-04a5-44de-b03d-11db8f36db80">

- Blue bars dominate at higher values (e.g., closer to 60-72 months). This means customers who have been with the company for a long time are more likely to stay.
- Red bars are higher at lower values (closer to 0-10 months). This shows that people who recently joined are more likely to leave.

In all of this, it means that customers are more likely to churn (leave) early in their tenure. Long-term customers tend to stay loyal.


2) For the MonthlyCharges (How much customers pay per month) visual:

<img width="297" alt="Monthly Feature numerical" src="https://github.com/user-attachments/assets/e59b884d-622c-4e0f-970b-6ebccf3a5223">

- The red bars are higher for larger MonthlyCharges (around $70-$120). This means customers who pay more each month are more likely to leave.
- The blue bars are spread across all charges, but they’re more concentrated in the middle range (around $30-$80).
This means that high monthly bills might be a reason why customers leave. Offering discounts or lower plans could help keep them.


3) In the TotalCharges (How much customers have paid overall) visual:
   
<img width="272" alt="totalcharges numerical feature" src="https://github.com/user-attachments/assets/dea12928-04a3-4d73-82ba-800f2f10e875">

- Blue bars are much higher for larger values (above $2000). This shows that customers who have paid more overall (likely those with long tenures) are more likely to stay.
- Red bars are higher for smaller values (closer to $0-$1000). This matches the tenure observation: newer customers (who haven’t paid much yet) are more likely to leave.
What this tells us is that customers who haven’t been with the company long or haven’t invested much money are more likely to churn.

We shall now move to the second feature distribution

### Categorical feature distribution

The charts here helps us visualize and understand customer demographics. By looking at how the groups are distributed.

<img width="717" alt="First" src="https://github.com/user-attachments/assets/3651376f-44fd-412c-b310-4d85e5dbfe98">

We shall explain every categorical feature 
1) Gender
   
The chart for gender shows:
Almost equal numbers of males and females in the data.
This means gender is balanced, so it probably won't strongly affect whether a customer leaves or stays.

2) SeniorCitizen
   
This chart shows:
Most customers are not senior citizens (0).
A smaller number of customers are senior citizens (1).

3) Partner
   
This chart shows:
About the same number of customers with a partner (Yes) and without a partner (No).

4) Dependents

This chart shows:
Many customers do not have dependents as the bar for No is much higher.
Fewer customers have dependents (Yes).

 <img width="844" alt="second" src="https://github.com/user-attachments/assets/4fd84d95-08a8-4d0f-af50-1bfce1041a93">

5) PhoneService

This chart shows:
Almost everyone has phone service, while a small group doesn’t.

6) MultipleLines

This chart shows:
Most people either have one phone line or multiple, and some don’t use phone service.

7) InternetService

This chart shows the type of internet people use:
"Fiber optic" is the most common, followed by DSL, and then people without internet.

8) OnlineSecurity

This chart shows:
Many people don’t have online security, while others do, and some don’t have internet service at all.

<img width="854" alt="Third" src="https://github.com/user-attachments/assets/e732a14c-2998-4963-a823-aff0c81b03ac">

9) OnlineBackup

This chart shows:
Many people don’t use online backup, while others do.

10) DeviceProtection

This chart shows:
Similar to OnlineBackup—many don’t have device protection.

11) TechSupport

This chart shows:
A lot of people don’t use tech support services.

12) StreamingTV

This chart shows:
Most people either have it or don't, while a smaller group doesn't have internet at all.

<img width="837" alt="fourth" src="https://github.com/user-attachments/assets/1f207b91-f27c-4905-9bcb-1f63649bdbb8">


13) StreamingMovies

This chart shows how many people have movie streaming services:
Most people either have it or don't, while a smaller group doesn't have internet at all.

14) Contract

Shows the type of contracts people have:
A lot of people are on "Month-to-month" contracts, while fewer have yearly contracts.

15) PaperlessBilling

Shows whether people use online billing:
More people use paperless billing than those who don’t.

16) PaymentMethod

Shows the payment methods:
Electronic checks are the most common, while other methods like mailed checks and automatic payments are less frequent.


Next, we examined categorical features in relation to target variable, 

<img width="803" alt="Categorical distribution in relation to churn" src="https://github.com/user-attachments/assets/609508ca-b2d4-47dc-a420-b171ee73fb5c">

However, we did this only for contract feature. 
1. Left Plot (Not Churned):
 - Most customers who didn’t churn have Month-to-Month contracts.
 - Many have Two-Year contracts, followed by One-Year contracts.

2. Right Plot (Churned):
 - A huge majority of churned customers had Month-to-Month contracts.
- Very few churned customers had One-Year or Two-Year contracts.
***What this means is that users who have a month-to-month contract are more likely to churn than users with long term contracts.***


We proceeded to the third feature distribution

### Target variable distribution. 

<img width="519" alt="Target feature" src="https://github.com/user-attachments/assets/b5843a35-e491-4ba4-838e-d8c10c687fa1">

Target variable distribution shows that we are dealing with an imbalanced problem as there are many more non-churned as compare to churned users. The model would achieve high accuracy as it would mostly predict majority class - users who didn't churn in our example.
Few things we can do to minimize the influence of imbalanced dataset:
- Resample data,
- Collect more samples,
- Use precision and recall as accuracy metrics
To minimize the influence of imbalance dataset, we go for the last option since we could not collect more samples.
Went further to check and remove outliers and the result showed that there are no outliers.

# Data Cleaning and Transformation
We dropped the “customer id” since it was not going to help us in our prediction and then moved on to perform one-hot encoding on our c categorical features which evidently expanded the number of our columns from 21 to 31 
We went over to feature scaling so as to prepare the data by splitting it into training and testing sets.

# Model Training

## Prediction using Logical Regression
Imported the logistic regression from the sklearn.linear model

Moving further, we ran the logistic regression model prediction

<img width="392" alt="Logical Regression analysis" src="https://github.com/user-attachments/assets/29b2e165-50ad-40e3-82e3-faecc39ede28">


Non-Churn (0)
  - Precision: 84% of predictions for Class 0 are correct.
  - Recall: 90% of actual Class 0 cases are predicted correctly.
  - F1-Score: 87% is the overall score for this class.
  - Support: There are **1557 samples** of Class 0.

Churn (1)
  - Precision: 65% of predictions for Class 1 are correct.
  - Recall: Only 53% of actual Class 1 cases are predicted correctly.
  - F1-Score: The overall score for this class is 58%.
  - Support: There are **556 samples** of Class 1.

For our Overall Metrics: 
- Accuracy: 80% of all predictions are correct.
- Macro Avg: Average metrics across both classes (unweighted).
  - Precision: 74%
  - Recall: 71%
  - F1-Score: 72%
- Weighted Avg: Average metrics across both classes, weighted by their support.
  - Precision: 79%
  - Recall: 80%
  - F1-Score: 79%


We can move on to visualize this 

<img width="773" alt="Regression bar chart" src="https://github.com/user-attachments/assets/0defb191-9105-4142-afcc-e461fb0175e7">


Moving on, we used the confusion matrix to predict the model. It tells us
ACTUAL VS PREDICTED as can be seen below:

<img width="512" alt="Logical actual" src="https://github.com/user-attachments/assets/6bf77237-ddeb-4dbb-bd3c-a4bd26d6499e">


from the image, 1397 customers who are not churning and have been predicted as not churning also, then there are 160 who are not churned but predicted as churned. 263 are actually churned and have been predicted as not churned, then 293 are churned and have been predicted as churned.

Comparing the trained dataset with the test dataset, we can notice that the different is just slight and so can be recommended.
for better understanding, see the image below:

<img width="218" alt="Regression test" src="https://github.com/user-attachments/assets/97e7664f-acbc-4d7a-bce0-3ba47fd8b530">

## Support Vector Classifier

we can understand the confusion metrix clearly  through the visual

<img width="411" alt="Vector actual vs predicted" src="https://github.com/user-attachments/assets/0ab562c9-4474-4389-99ad-b001adc517f6">

We can see that 1427 customers who are not churning and have been predicted as not churning also, then there are 130 who are not churned but predicted as churned. 292 are actually churned and have been predicted as not churned, then 264 are churned and have been predicted as churned.


So moving further, we  can see that the trained dataset has more score than the test dataset as can be seen below:

<img width="239" alt="Vector Train" src="https://github.com/user-attachments/assets/dc1eccbe-24d0-4b71-a4f9-056d14df6125">


## Precision using decision tree classifier.
After training and testing the model, we can understand better with the visual 

<img width="400" alt="Decision Tree chart" src="https://github.com/user-attachments/assets/6d16f5c8-bfaf-46ce-ab84-04ea6c5bff1b">


The visual shows that 1267 customers who are not churning and have been predicted as not churning also, then there are 290 who are not churned but predicted as churned. 284 customers who are actually churned and have been predicted as not churned, then 272 are churned and have been predicted as churned.

So we can check the score of the test and train data 

<img width="232" alt="tree test" src="https://github.com/user-attachments/assets/b30c1010-adcb-4b0b-95ec-bdd545503861">

Notice that the difference is much and so, I would not recoomend using the DECISION TREE CLASSIFIER.


## KNN Classifier
After training and testing the model. I checked for the error rate, then moved on to run the confusion matrix

<img width="520" alt="KNN Graph" src="https://github.com/user-attachments/assets/7a0f6171-fa2a-479d-bdde-a8e3d87f22e4">

Here are the takes from the visual:

1369 customers who are not churning and have been predicted as not churning.
188 who are not churned have been predicted as churned. 
250 customers who are actually churned have been predicted as not churned.
306 are churned and have been predicted as churned.

We checked the score:

<img width="199" alt="KNN Test" src="https://github.com/user-attachments/assets/de12e503-798c-4c69-baad-f55d853e783e">

Since its close, it can work.


# Conclusion
This detailed analysis integrates data exploration, preparation, and predictive modeling to understand and mitigate customer churn. By applying these models, the company can proactively reach out to at-risk customers and implement retention strategies effectively, enhancing customer satisfaction and stabilizing revenue. ​​


# Recommendations for the Company

Based on the findings and model predictions, the company should consider:
- Enhancing Customer Engagement: Focus on segments identified as high-risk, such as customers on month-to-month contracts or those with specific service plans.
- Adjust Service Contracts: Since month-to-month contracts have higher churn as indicated by the analysis. Offering discounts for long-term contracts can help retain customers.
- Tailored Offers: Providing personalized retention offers for customers at risk based on the model's insights.
- Continuous Model Monitoring: Implement a system to continuously feed new data into the predictive model. That is, regularly update the model with new customer data to ensure it adapts to changing customer behavior and maintains accuracy.
- Improve Customer Support: Address pain points highlighted by the model, such as billing issues or technical support, to increase satisfaction.
- Focus on new customers or high-paying customers (they are at a higher risk of leaving).
- Offer special deals to lower churn in groups at risk (like reducing prices for customers with high monthly charges).
- Use targeted communication to keep high-value customers engaged.
- Introduce loyalty programs or incentives for customers with higher churn risk.


## Thanks for Reading
