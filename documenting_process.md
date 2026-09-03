The first step before training any ML model is to prepare the data appropriately.


Why?
    We know the credit card fraud detection is a classification problem.
    So naturally my first intuition is to use logistic regression.
    Logistic regression requires my data to have no extreme outliers and no high multicollinearity 
    It also requires the data to be of good quality as null/NaN values crash the algorithm

Therefore, we perform Exploratory Data Analysis.



**EXPLORATORY DATA ANALYSIS**



EDA is divided into four parts:
-Data Quality Inspection.
-Univariate Analysis.
-Bivariate & Multicollinearity Analysis.
-Feature Engineering.


1.) Data Quality Inspection.
Since I have an open-sourced data quality agent that detects DQ issues on datasets, I ran this agent on my data first. The only issue it told me existed was the existence of duplicate values. So this finding indicated that I must check this manually & remove the duplicates.
I must also check for other potential DQ issues that might exist manually as the agent is obviously prone to error.
Why is having duplicated rows bad for example?
    Say if I have 2 identical rows, one of them in the training set and one of them in the test set after splitting.
    While training, the model learns the pattern in the first example. Then when it encounters the same row in the test split, it will give the right answer from its memory. We don't want our model to do this as this way, the accuracy among other metrics will be highly inflated.

What did I do?
-Check the number of rows and columns of my table using df.shape()
-df.head() showed me what values the columns should represent
-df.info() told me which datatypes the columns actually store, I needed to verify that they were all numeric, which they were.
 If they weren't, I'd convert the datatypes from whatever it is to numeric.
-df.isnull().sum() showed me the null values in each of the columns ->no nulls returned
-df.duplicated().sum() returned 1081, which meant there were those many duplicates in my dataset
-df.drop_duplicates() dropped the duplicates & then I verified that they were actually dropped


2.) Univariate Analysis.
In this phase, we look at each column in complete isolation, one at a time.
Our objective here is to try and understand the nuances observed in each column individually.
The columns can either be target variables (Y AKA output) or features (X).
-For features, we look at each of them one by one, plot their distributions and check for things like whether they are normalized and if they have outliers for      example. 
-For target variables, we wanna try and see the frequency distributions of all the outputs and this will help us come up with suitable metrics to evaluate our model.

-Target Variables:
Ran df['Class'].value_counts() to get an idea of the number of classified Legit vs Fraud rows.
Observed that around 99.83% of the rows are Legit and the remainder Fraud.
This was even plotted visually.
This led us to believe that accuracy would be a useless metric to evaluate our model.
Therefore we must rely on Precision, Recall, F1-Score, and PR-AUC instead.

-Features:
a.) Verifying the PCA transformation for ALL the features:
    -Dataset claims that V1 to V28 are Principal Components resulting from a PCA transformation. Why is it that way?
     Because credit card information is proprietary and if non numeric cols are represented numerically using PCA, it hides sensitive information.
    -Thus our first objective is to ensure that PCA was actually performed properly. 
    -So we looked for 3 signals: 
        Zero Mean: The means of each of the features must be around 0.
        Decreasing Variance from V1 to V28: self-explanatory.
        Zero Correlation: zero correlation between different columns and correlation of 1 for same columns (obvious).
                          so we draw a correlation matrix and hope to observe only the diagonals having elements 1 and the rest around 0.
        None of these will be exactly 0 or 1 due to our data cleaning process.
    We ended up getting what we were looking for and therefore PCA transformation was verified.
b.) Looking for outliers in each feature:
    -We take min, max, mean and std for each feature.
    -Find the z score for the min and max value. z score indicates the number of standard deviations a number is from the mean.
    -Highest z score columns indicate more outliers. Outliers MIGHT be a sign of frauds. So this is handy information.
    We found that V28, 7, 5, 23, 27 have the largest z scores and this might be a signal for fraudulent behaviour.



3.) Bivariate & Multicollinearity Analysis.
Here, features are compared against each other and against the target variable. 
Thus there are two parts to it:


-Bivariate Analysis (feature vs target)
We look at the relationship between each of the features vs target variable (Class).
Our objective is to find the features with the most Predictive Power.
i.e which features cleanly seperate legitimate transactions from fraudulent ones?

a.) Amount
 We first decided to check if Amount was a key measure that could be used to detect fraud.
 We box-plotted it initially, but due to the large number of outliers, the box got squished and wasn't visible.
 So we applied a log transformation on the data and box-plotted that.
 These were the observations:
 -median & 75 percentile are similar (massive overlap) 
   An ML algorithm will struggle here because if it sees a transaction in this overlapping region, 
   it won't be able to differentiate between the two classes using Amount as the parameter.
 -Legit has bigger outliers
   This means that huge transactions are more linked to Legit than Fraud.
 -25% percentile is much lower in fraud than legit.
   This means transactions involving lesser amounts are more linked to Fraud than Legit. 
   Maybe fraudsters test out their strategies with lesser amounts of money initially to minimize their losses.
 
 FINAL CONCLUSION for Amount:
 While there are tiny nuances, the massive overlap between the two boxes means that Amount is a weak standalone predictor. 
 So we cannot rely on it heavily to catch fraud. 

b.) Evaluating PCA features (V1-28)
 df.corr() AKA pearson correlation was designed to measure how two continuous numbers move together in a straight line.
 One party is continuous (PCA features), but the other (Class AKA Legit or Fraud) isn't. Therefore, we can't use df.corr().
 Another reason why we can't: df.corr() is extremely sensitive to outliers and they can easily skew the correlation score, 
 making it look artificially high or low. 

 So what's the fix?
 ->Using Standardized Mean Difference. (AKA Effect Size)
 This parameter measures the distance between the centres AKA medians of the Fraud to the Legit data.

 Effect size=(Median of Fraud - Median of Legit)/Standard Deviation

 Why divide by std? Some cols have huge spreads whereas others have a small spread. Div by std helps standardize the measurement 
 of the gap in a universal unit.

 What do we infer from this?
 ->Effect size measures the physical gap between the two boxes on a boxplot.
 ->Thus, larger the gap, further the Fraud and Legit boxes are from each other. 
 ->Hence, making it easier for our ML algorithm to differentiate and draw a boundary between them & vice versa.

 So we found the effect size and chose the top 5 columns with the highest effect size and box plotted them.
 We noticed that the box plots of Legit vs Fraud do not overlap and will therefore be easily differentiable by our ML algorithm later on.
 Columns chosen: V10, 12, 14, 16, 17.


-Multicollinearity Analysis (feature vs feature)
We know that for logistic regresion to work, there cannot be high multicollinearity.
What does that mean? 
->If there is high correlation between two variables, then using them both is highly redundant. This confuses LR and confuses the model.
So we gotta check for Pearson's correlation between all combinations of all the features. 
Why pearson's? 
->Because all the variables are numeric & continuous.

The funny thing is, while checking for PCA, we already created a correlation matrix for all the PCA variables (V1-V28).
So now, the only thing left to do is to do the same including the Amount and Time columns.
We are hoping to see a diaganol of 1s and the rest of the elements not close to 1.

Observations:
Since amount and time didn't undergo PCA, some of their correlation values are above 0. 
However, we cannot see severe amounts of multicollinearity anywhere and can therefore continue with this data for our Logistic Regression model.



4.) Feature Engineering:

GENERAL EXTREMELY IMPORTANT POINTERS:

Feature Engineering is the process of creating new columns from existing data to help the machine learning model find patterns. 
However, modern algorithms (like XGBoost) are incredibly smart, meaning we must be careful not to "over-engineer" and accidentally destroy data. 

The Golden Rule: You only engineer a feature when the model is mathematically blind to a pattern that you, as a human, know exists. 

Some of the guidelines we follow:

**1. Fix "Unreadable" Formats**
Models only understand patterns in numbers. If a format doesn't look like a repeating pattern, you must engineer it.
* *Example:* Our `Time` column is a linear counter of seconds (0 to 172,800). A model learns nothing from a number that never repeats. 
* *Solution:* We engineered `Time` into an **Hour of the Day (0-23)** feature. Now the model can easily read the cyclical pattern and learn if 
            fraud peaks at 3:00 AM. 

**2. Avoid "Binning" Continuous Variables**
We avoided grouping `Amount` into "High/Medium/Low" buckets. 
* *Why?* Binning destroys information. It forces the model to treat a $51 transaction and a $199 transaction as the exact same "Medium" event.
 Furthermore, tree-based models (like XGBoost) mathematically calculate the absolute perfect split-points during training anyway. 
 Manually binning data usually just makes the model worse.

**3. Avoid "Multiplication/Interaction" Features**
We avoided creating features like `Amount * Hour` (Amount spent at a specific hour).
* *Why?* Tree models are giant chains of "IF/THEN" statements. 
They naturally replicate multiplication and "AND" logic on their own by learning: *"IF Hour is 3 AND Amount is > 500 THEN Fraud"*. 
Spoon-feeding it an interacted column provides no new information and risks introducing dangerous multicollinearity to our Logistic Regression model.

**4. DO Engineer "Division/Ratio" Features**
While we didn't have the columns for it in this dataset, engineering ratios is highly recommended in Data Science.
* *Why?* Tree algorithms are terrible at division. An "IF" statement cannot calculate a fraction. 
A model will look at a `$500k` house (5000 sqft) and a `$100k` house (1000 sqft) as wildly different numbers, 
completely missing that they share the exact same `$100/sqft` ratio. Because models cannot divide across rows, you must engineer ratios for them.

So what did we do?
-> We used the Time column to create an hour_of_day column.
-> Since our data spans across two days, we can maintain this metric easily.
-> plotted KDE Plot, which told us that fraudulent transactions peak around hour 2.

The KDE Plot calculates Density (the proportion of transactions that belong to a class).

Note a couple of very important observations:
-At 11am, 80 frauds out of 15,000 transactions are observed.
-At 2am, 40 frauds out of 240 transactions are observed.
-This tells us that even though the absolute number of frauds is highest at 11am, there is a much highest probability of a fraudulent transaction happening
 at 2am.
-Our ML model learns based on probability. Therefore we only care about the probability of getting scammed. Hence, we've used a KDE Plot to observe this info.



EDA COMPLETE!


EDA was for the Human to understand the data.
The next step is Data Preprocessing. 
Here, the machine understands the data.



**DATA PREPROCESSING**


Data Preprocessing is entirely about formatting the data so it doesn't crash the algorithm. 
Machine learning models are just dumb calculators & are incredibly picky about how numbers are fed into them. 
Preprocessing is the rigid, mathematical formatting of the data.

So what do we gotta do?
->Splitting the data into Train and Test sets (so we can test the model later without cheating).
->Scaling the data (forcing large numbers like $25,000 into a small mathematical range like 0.0 to 1.0 so the math equations don't break).
->Encoding text (if we had a column that said "New York", preprocessing converts it to 1 because models can't read letters).

Result: The data is mathematically sterile and ready to be plugged into model.fit().


1.) Train-Test split:
Decided to split the data over the 2 days in 50% each, for the training and test sets both.
The reason being, one of our biggest differentiators between fraud and legit labels is the hours of the day
i.e at 2am the probability of fraud transactions spikes.
Therefore since we've got 2 days of data, we wanna split our data equally over the training and test set to see if it can infer the correct solutions 
in the test set.

2.) Scaling:
The PCA features (V1-28) are already perfectly scaled (centred around 0).
But our Amount col has massive outliers and Hour col goes upto 23.
If we don't scale these down, the Logistic Regression model will think Amount is 25,000x more important than the V columns when it assigns weight. 
(25k is the highest Amount)
This is why we scale.

PREREQUISIES:
-fit: Learning Phase
 When we run .fit(X_train), the scaler looks at the data and calculates the Median and the IQR. It DOES NOT change anything. 
 It basically "fits" itself to the patterms of the training data.
-transform: Action Phase
 When we run .transform(X_train), the scaler uses the median and IQR it learned from the "fit" phase and applies (value-median)/IQR to all the rows.
 The old rows are now overwritten with the newly scaled numbers.
-fit_transform: Learning + Action
 When we run .fit_transform(), it calculates the Median & IQR, memorizes it, and immediately applies it to scale the numbers.
-ColumnTransformer: Python library used for scaling
 Whatever this library does can easily be performed using simple maths. But the issue is, the mathematical way will be a nightmare for deployment on Streamlit/anything else.
 When we launch our Streamlit dashboard later on, we would have to manually rewrite these exact slicing steps for every new transaction that comes in.
 ColumnTransformer is the industry standard that automates both, the scaling as well as the deployment part. That's why we use it.
 RobustScaler() is actually the library we use for scaling and ColumnTransformer brings everything together.
 It also completely prevents data leakage if done right.

**GOLDEN RULE OF SCALING: (how do we avoid data leakage?)**
**NEVER EVER fit on the Test data.**
Imagine if we used .fit() on the entire data (train+test). The scaler would calculate a median that includes the test data.
Then, when we train our model on the scaled data, the model secretly learns hints about the test data.
This is Data Leakage and leads to inflated accuracy scores.
**Correct way:**
.fit_transform(X_train): Learns median & IQR from the training data and scales the training data using those numbers.
.transform(X_test): Scale the test data using the median & IQR obtained from the training set.

So this is exactly what we did in order to scale our data.

Once we scaled it, we converted the outputed array to a dataframe to add our column names and row numbers to it again.

Scikit-learn tools like ColumnTransformer strip away all the column names (like 'V1', 'Amount', etc) and the row numbers.
It spits out a raw Numpy array, which is just an anonymous block of numbers. We can't see which column is which.
While the ML algorithms are perfectly capable of training on this array, we still need the col names and row numbers in the explainability (SHAP) phase of our project.
If we don't have col names, SHAP will just say "Feature 0 is most important".
Instead, we want it to say "Hour is the most important". (example)

3.) Encoding:
Not required as all columns are numeric.



DATA PREPROCESSING COMPLETE



**Model Training and Advanced Validation**


