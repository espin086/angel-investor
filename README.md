# angel-investor
This is an artificial intelligence tool to rate the probability of a start-up succeeding in the market. 

## Domain Background
---
Start-up financing is a topic that has been researched extensively in the fields of finance, management, and economics. Financing of start-ups differs from traditional financing of companies in terms of risk and valuation methodologies. These topics have been explored in these highly-cited academic studies: 

* Yang, C., Bossink, B. and Peverelli, P., 2017. [High-tech start-up firm survival originating from a combined use of internal resources.](https://personal.vu.nl/p.j.peverelli/ChunBossinkPeverelli.pdf) Small Business Economics, 49(4), pp.799-824.
* Cole, R.A. and Sokolyk, T., 2018. [Debt financing, survival, and growth of start-up firms.](https://rebelcole.com/PDF/Cole-Sokolyk.JCF.2017.pdf) Journal of Corporate Finance, 50, pp.609-625.
* Fuertes-Callén, Y., Cuellar-Fernández, B. and Serrano-Cinca, C., 2020. [Predicting startup survival using first years financial statements.](https://www.researchgate.net/profile/Carlos-Serrano-Cinca/publication/343566221_Predicting_startup_survival_using_first_years_financial_statements/links/6024f89f92851c4ed5639c6a/Predicting-startup-survival-using-first-years-financial-statements.pdf) Journal of Small Business Management, pp.1-37.
* Åstebro, T. and Bernhardt, I., 2003. [Start-up financing, owner characteristics, and survival.](https://d1wqtxts1xzle7.cloudfront.net/49879491/TM___JK_Start-up_Financing_manuscript_2010_11_04.pdf?1477479846=&response-content-disposition=inline%3B+filename%3DStart_up_Financing_in_the_Age_of_Globali.pdf&Expires=1626899842&Signature=cjj9vaq~lLG6Kv8jJr~I43QPyteiNRxrY-rmBGAoF5u5UjoWUX3HNKwrtCM2xwGTn8JacvO6Jd~BvuN~xdhb6Vq4fbCsKR9HVPnObiAj2DLRceKK3aJ-7uK2CuS9llLYZ666o3EERwRSlzKJk8OUTW5c9JBQ~AWP~DNOjMxylgSon6MHPMCHOktlxhlvRBjcd0g7lHcFUqYq2WO5rXQUwpl8~qgbqeqV-hiLCnxLIsHxDXJx8C2yqu0NW7vIoXeWW0snfEZ-Rw52Z781SWHwJUNo5JQjxzjNfXQ~eqQF1bJRkZh9s34PyG7v~ziwQ3~~gTnMUrHRTkPI0nOWtyOQhg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) Journal of Economics and Business, 55(4), pp.303-319.
* Gartner, W., Starr, J. and Bhat, S., 1999. [Predicting new venture survival: an analysis of “anatomy of a start-up.”](https://www.sciencedirect.com/science/article/pii/S0883902697000633) cases from Inc. Magazine. Journal of Business venturing, 14(2), pp.215-232.


## Problem Statement
---
The people who fund early-stage start-ups often take on considerable risks and are known as Angel Investors and Venture Capitalists. Angel Investors and Venture Capitalists use their business knowledge, judgement of character, and some early quantiative information to judge the growth potential of a start-up. Given the high-risk/high-reward nature of start-up financing the more information an investor has on a start-up the better.   

## Datasets and Inputs
---
The datasets used in the project include data on over 40,000 start-ups from around the world. The source of the dataset is Kaggle.com, a platform that hosts Machine Learning and Artificial Intelligence competitions and data. The data originated from Crunchbase. 

* [Link to Data Set in Kaggle](https://www.kaggle.com/arindam235/startup-investments-crunchbase)


Labels: the label for this dataset indicates whether a start-up:
* Is still operating (in-business)
* Closed down and went out of business
* Was acquired by a larger company

*note: that the data are unbalanced as the majority of the companies are still operating 80+% and only about 9% have failed. This is contrary to what we see in real life, where the majority of start-ups fail. This will be dealth with using various sampling techniques like SMOTE and oversampling the minority class. 


* Datasets and their variables

     * **Companies (dataset):**
       * permalink
       * (primary)name
       * homepage_url
       * category_list
       * market
       * funding_total_usd
       * country_code
       * state_code
       * region
       * (label) status
       * region
       * city
       * funding_round
       * founded_at
       * founding_year
       * founding_month
       * funding_quarter
       * first_funding_at
       * last_funding_at
       
     * **Rounds (dataset):**
       * company_permalink
       * (primary) company_name
       * company_category_list
       * company_market
       * company_country_code
       * company_state_code
       * company_region
       * company_city
       * funding_round_permalink
       * funding_round_type
       * funding_round_code
       * funded_at
       * funded_month
       * funded_quarter
       * funded_year
       * raised_amount_usd
      
     * **Investments (dataset):**
       * company_permalink
       * (primary) company_name
       * company_category_list
       * company_market
       * company_country_code
       * company_state_code
       * company_region
       * company_city
       * investor_permalink
       * investor_name
       * investor_category_list
       * investor_market
       * investor_country_code
       * investor_state_code
       * investor_region
       * investor_city
       * funding_round_permalink
       * funding_round_type
       * funding_round_code
       * funded_at
       * funded_month
       * funded_quarter
       * funded_year
       * raised_amount_usd

    * **Acquisitions (dataset):**
       * company_permalink
       * (primary) company_name
       * company_category_list
       * company_market
       * company_country_code
       * company_state_code
       * company_region
       * company_city
       * acquirer_permalink
       * acquirer_name
       * acquirer_category_list
       * acquirer_market
       * acquirer_country_code
       * acquirer_state_code
       * acquirer_region
       * acquirer_city
       * acquired_at
       * acquired_month
       * acquired_quarter
       * acquired_year
       * price_amount
       * price_currency_code



## Solution Statement
---
This project's goal is to build a solution that reduces the risk of investing in a start-up by creating a predictive model (machine learning model) of start-up success as defined by not-failing within a set number of years. 

## A Benchmark Model
---
It is estimated that about 90% of start-ups fail with 22% failing in their first year, 30% in the second year, and 50% in their fifth year. [source](https://www.investopedia.com/articles/personal-finance/040915/how-many-startups-fail-and-why.asp). The benchmark model will be based on random selection of start-ups versus the machine learning algorithms to see what type of improvement may be seen.

## Evaluation Metrics
---
The evaluation metrics will be Precision (True Positive / (True Positive + False Positive)) and ROC-AUC. The goal is to reduce the likelihood of missing out on a really great start-up because one didn't invest in it. Thus the goal is to reduce the number of 'false negatives' when asking the question, 'does this start-up have a chance of being acquired?'.

## Project Design
---
The project will be designed in the following way. 

* Data will be explored and cleaned including: 
    * Scaling and normalizing the data
    * An assessment will be made to drop missing values or to use imputation to fill in missing values
    * Outliers in the data will be detected using a Random Cut Forest Algorithm, careful attention will be made in how to handle the outliers. If they are data errors then they will be deleted, but in this case an outlier may be a start-up that turns out to be Facebook or Google and we would not want to drop them as those are the kinds of start-ups that angel investors would NOT want to miss. 
* Exploratory analysis will determine the right variables to include in the model, resulting in: 
     * Testing of variable reduction techniques like Principal Component Analysis
     * Dropping of redundant variables
* Feature engineering will extract relevant information to improve model performance
* A model will be trained on a training set and evaluated on a test set, several models will be attemped to classify the changes of a start-up succeeding: 
     * K-Nearest Neighbors 
     * Linear Learner
     * XGBoost Algorithm
* A production endpoint accessible via API (API Gateway/Lambda) will be created to deploy the model

