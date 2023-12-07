# Customer Segmentation and Predictive Analysis

## How to run

1. **Download the Repository**: Clone or download the repository from GitHub to your local machine.

2. **Install Requirements**: Run `pip install -r requirements.txt` to install the required Python packages listed in `requirements.txt`.

3. **Run the Main Script**: Execute the main script of the project by running `python main.py`.

## Motivation
The motivation behind this project is to apply machine learning and data analysis techniques to understand customers behaviour better. This was done by segmenting customers into meaningful groups, understanding each customer clusters (3 in total), and predicting customer responses to the latest marketing campaign based on those clusters.

**Dataset:** Customer Personality Analysis

**Link:** https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data

**NOTE:** Dataset is considered to be data collected over the period of 2 years.

## Project Overview

### 1. Data Overview and Preprocessing
- Dataset: 2240 instances, 29 features.
- Steps: Handling missing values, duplicates, outliers; categorical data encoding; dataset standardization.

### 2. Customer Segmentation
- Method: K-Means clustering, supported by PCA.
- Result: Identification of three distinct customer groups.

### 3. Customer Profiling
- Analysis: Detailed profiling based on demographics, income, and spending patterns across customer groups.

### 4. Predictive Model Analysis
- Approach: XGBoost algorithm for campaign responsiveness.
- Outcome: Enhanced understanding of varied customer responses to marketing campaigns.

### Summary of Findings

- Identified three unique customer segments with distinct characteristics.
- Most decisive differences between customer were income, spending, and age.
- Recall is improved when predicting customer response to a campaign by separating into customer clusters. Though not by much.
- There could be a benefit from using KNN or a clustering method to first assign a customer into a cluster before predicting customer's response to a campaign.

## 1. Data Overview and Preprocessing 

### A. The dataset contains 2240 instances and 29 features

Attributes

People

    ID: Customer's unique identifier
    Year_Birth: Customer's birth year
    Education: Customer's education level
    Marital_Status: Customer's marital status
    Income: Customer's yearly household income
    Kidhome: Number of children in customer's household
    Teenhome: Number of teenagers in customer's household
    Dt_Customer: Date of customer's enrollment with the company
    Recency: Number of days since customer's last purchase
    Complain: 1 if the customer complained in the last 2 years, 0 otherwise

Products

    MntWines: Amount spent on wine in last 2 years
    MntFruits: Amount spent on fruits in last 2 years
    MntMeatProducts: Amount spent on meat in last 2 years
    MntFishProducts: Amount spent on fish in last 2 years
    MntSweetProducts: Amount spent on sweets in last 2 years
    MntGoldProds: Amount spent on gold in last 2 years

Promotion

    NumDealsPurchases: Number of purchases made with a discount
    AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
    AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
    AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
    AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
    AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
    Response: 1 if customer accepted the offer in the last campaign, 0 otherwise

Place

    NumWebPurchases: Number of purchases made through the company’s website
    NumCatalogPurchases: Number of purchases made using a catalogue
    NumStorePurchases: Number of purchases made directly in stores
    NumWebVisitsMonth: Number of visits to company’s website in the last month

### B. Features were renamed for readibility purposes

The DataFrame columns were renamed as follows:

- `Kidhome` -> `Kid`
- `Teenhome` -> `Teen`
- `MntWines` -> `Wine`
- `MntFruits` -> `Fruit`
- `MntMeatProducts` -> `Meat`
- `MntFishProducts` -> `Fish`
- `MntSweetProducts` -> `Sweet`
- `MntGoldProds` -> `Gold`
- `NumDealsPurchases` -> `Deals Purchases`
- `NumWebPurchases` -> `Web Purchases`
- `NumCatalogPurchases` -> `Catalog Purchases`
- `NumStorePurchases` -> `Store Purchases`
- `NumWebVisitsMonth` -> `Web Visits Month`
- `AcceptedCmp1` -> `Campaign 1`
- `AcceptedCmp2` -> `Campaign 2`
- `AcceptedCmp3` -> `Campaign 3`
- `AcceptedCmp4` -> `Campaign 4`
- `AcceptedCmp5` -> `Campaign 5`
- `Response` -> `Latest Campaign`

#### C. Handling Missing Values

The dataset had a minimal number of missing entries, with only 24 fields in the "Income" attribute missing. These were imputed using the KNN algorithm, which predicts missing values using the nearest neighbors' data points.

![Missing Values Imputation](https://github.com/eliasright/Customer/assets/151723828/e05ed7e8-35ff-4ade-a8a2-5c2780b98163)

#### D. Duplicate Values

The dataset contains 174 duplication groups. Each groups have the same values. Only 1 record were kept from these groups. Through further inspection, there exist another duplication groups where the only difference is the feature "Latest Campaign". It is uncertainty why this is the case but most likely those with Latest Campaign as 1 is the most recent record as this is a binary showing if a customer engaged with the last campaign. For all of these groups, I kept only the record with 1. (Assuming 0 were not yet updatet).

![Duplicate Values](https://github.com/eliasright/Customer/assets/151723828/3cec6824-e8d4-4aab-90a6-ad07281c2958)

#### E. Replacing Values

Assuming a minimum data recording date based on the "Dt_Customer" (Date customer joined the company membership programmne) plus "Recency" (Number of days since customer last purchased something). Using the follow date 04/10/2014:
- The "Year_Birth" was converted to "Age"
- "Dt_Customer" was replaced with "Membership Length"

#### F. Marital Status Categorical

In the marital status there were non-standard inputs such as "Alone", "Absurd", and "YOLO". They are considered false data entry, and hence were cleaned by categorizing them as "Single", resulting in a revised set of categories: Married, Together, Single, Divorced, Widow.

![Marital Status Distribution](https://github.com/eliasright/Customer/assets/151723828/322694a1-d6bc-49aa-85b6-92e0ff6a13b6)

#### G. Encoding Categorical Data

The "Marital Status" and "Education" categorical data were encoded using One Hot Encoding, with prefixes "Marital" and "Edu", respectively. Also dropping the first column since it is redundant.

![Education Distribution](https://github.com/eliasright/Customer/assets/151723828/24364bcc-f853-4766-9cd9-2b54d246ad32)

#### H. Outliers

For "Income", outliers were addressed differently. Extreme values near the whisker (high end) were retained since it is highly plausible (true outliers). While an improbable entry (annual salary of $666,666) was adjusted using KNN imputation to a more appropriate value. The instance was not dropped because the other features were not outlier and deemed probable and likely correct.

![Income Outliers](https://github.com/eliasright/Customer/assets/151723828/304977f1-6ebb-426d-b91c-7384c1e5f6a5)

The "Age" outliers, suggesting people over the age of 100 with substantial income, were considered either inaccurate or just improbable. Even if it was true, it would still be an extreme outlier. The Birth_Year also suggested that 1900 and 1899 were likely a false data entry by the customer. These were also corrected with KNN imputation.

![Age Outliers](https://github.com/eliasright/Customer/assets/151723828/e00d8293-2620-4acd-9378-b9f425cedd79)

The overall dataset's measures of central tendency and dispersion were checked, and no values were deemed outliers based solely on deviation from the norm. These values were collected by the company, hence are likely to be correctly logged (except "Latest Campaign" that was dealth with in duplication process). The table below shows purchases based on categories and purchasing methods. Nothing notable.

![Purchases and Methods Overview](https://github.com/eliasright/Customer/assets/151723828/70a48efe-a552-4bc1-a851-f29598c586b3)

#### I. Dataset Standardization

Finally, a standardization process was applied to normalize the feature scales across the dataset. Choosing only those with a range above 3. 

![Before and After Transformation](https://github.com/eliasright/Customer/assets/151723828/e0638266-03a3-42af-96bd-01262d8779fa)

## 2. Customer Segmentation

### A. Determining the Optimal Number of Clusters

The process of identifying the optimal number of clusters for segmentation is tricky. Without domain knowledge, I utilized the Elbow Method. Mathematically determine each points diminishing rate of variance reduction with increasing cluster numbers. This analysis robustly suggests that the optimal number of clusters (K) is three for our customer data. Graph below shows the "Knee Point" and Distortion at each point. So 3 was chosen as the number of cluster to investigate.

![Knee Point Graph](https://github.com/eliasright/Customer/assets/151723828/7a3c9cf5-c25e-4e64-af8b-8cb95e70c024)

### B. Implementation of K-Means Clustering Algorithm

I used the K-Means clustering algorithm to segment the customer dataset. K-Means is known for its efficiency and clear clustering, making it suitable for processing large data volumes. It helps identify distinct customer groups.

In the clustering process, I excluded features related to campaign interactions (`Campaign 1` to `Campaign 5` and `Latest Campaign`) to avoid overfitting and bias in the predictive modeling. This exclusion ensures the segmentation results are unbiased and the predictions about customer engagement with `Latest Campaigns` are accurate.

To reduce the dataset's dimensionality with minimal information loss, I applied Principal Component Analysis (PCA). I found that seven principal components represent over 70% of the dataset's variability, reducing the original 28 features to seven and simplifying the dataset while preserving its interpretability. 

After applying PCA, I used the K-Means algorithm to form three clusters. I then applied PCA again for a three-dimensional visualization of the customer groups, providing a clear and informative view of the clustering.

![3D Clusters Visualization](https://github.com/eliasright/Customer/assets/151723828/0c6a2920-8cc7-49b7-bf7a-97e5efbddf80)

## 3. Customer Profiling

### A. Summary

I have profiled our customer base into three main clusters, each characterized by distinct features:

1. **Group 0 (1028 Customers)**
   - **Demographics**: Young, new to our brand, basic education, mainly single, some parents.
   - **Economics**: Lowest income and spending in the group, with some high-income exceptions.

2. **Group 1 (625 Customers)**
   - **Demographics**: Middle-aged, longer brand association, higher education, varied marital status, fewer parents.
   - **Economics**: Highest income and spending.

3. **Group 2 (587 Customers)**
   - **Demographics**: Oldest group, longest brand association, advanced degrees, more widowed/divorced, teenage children.
   - **Economics**: Moderate income and spending.

### Assumptions and Insights

- **Group 0**: Young, early career stage, basic education, low income and spending.
- **Group 1**: Financially stable, mature, higher education, fewer child responsibilities.
- **Group 2**: Older, financially prudent, advanced education, beyond intensive child-rearing stage.

### B. Membership 

![Membership_Length](https://github.com/eliasright/Customer/assets/151723828/23555a46-c447-4bff-98b7-15ccfb8dbf36)

The membership count for each cluster/group is as follows:
- **Group 0:** 1028
- **Group 1:** 625
- **Group 2:** 587

The length of membership escalates from Group 0 to Group 2, indicative of increasing customer loyalty or tenure.

### C. Age

![Age_Box_Whisker](https://github.com/eliasright/Customer/assets/151723828/0283707a-a7e8-46d6-ae35-35142fc2e9a3)

- **Group 0:** Characteristically younger with fewer age outliers.
- **Group 1:** Slightly older with a broader age dispersion.
- **Group 2:** The most senior with a moderate age range.

### D. Education

![Education_Distribution](https://github.com/eliasright/Customer/assets/151723828/eafa1d36-94ec-4c33-82db-be6daec7a01c)

The distribution of educational qualifications shows some variation among the groups:
- **Group 0:** The only group to include customers with a basic level of education.
- **Group 1:** Boasts the highest number of graduates.
- **Group 2:** Features a greater proportion of customers with master's and doctoral degrees, arguably the most academically credentialed.

### E. Marital Status

![Marital_Distribution](https://github.com/eliasright/Customer/assets/151723828/0b83754b-c1a7-4b57-96ad-f901584c377e)

Marital status distributions align somewhat with age demographics:
- **Single status** diminishes progressively across the groups.
- **Widowed status** increases with each subsequent group.
- **Divorced status** follows a similar upward trend.

### F. Parenthood (Teen and Kid)

![Parent_Distribution](https://github.com/eliasright/Customer/assets/151723828/281175a8-2c74-4a60-b451-0b0cfe539759)

- **Group 0:** Exhibits a higher percentage of households with children, either young or teenagers, or both.
- **Group 1:** Contrasts sharply with Group 0, having a substantial majority without children.
- **Group 2:** Noteworthy for the highest percentage of households with only teenagers.

### G. Income and Spending

#### Income

![Income_Box_Whisker](https://github.com/eliasright/Customer/assets/151723828/a04e4e7d-dcd2-42f9-acca-3c9876bfbc4d)

- **Group 0:** The lowest median income but with some extreme high earners.
- **Group 1:** The most affluent, with a median income far surpassing the other groups.
- **Group 2:** Earners of a moderate mean, positioned between Groups 0 and 1.

#### Spending

![Spending_Box_Whisker](https://github.com/eliasright/Customer/assets/151723828/74577b6c-f3a4-4a84-8f24-33ee9dd9c960)

- **Group 0:** The least spending, not entirely accounted for by shorter membership duration.
- **Group 1**: The most spending, mirroring its income pattern.
- **Group 2:** Median spending is about half that of Group 1.

#### Spending & Income

![Income_Spending](https://github.com/eliasright/Customer/assets/151723828/0a495eda-b7ce-41b0-852b-d094008d82f4)

A PCA visualization of spending against income reveals:
- **Group 0:** Closest to the origin, indicating the lowest income and spending.
- **Group 1:** Farthest from the origin, suggesting the highest income and spending.
- **Group 2:** Intermediate position on both income and spending, between Groups 0 and 1.
  
## 4. Predictive Model Analysis for "Latest Campaign"

The project focused on predictive modeling to forecast customer responses to our "Latest Campaign". The goal was to identify target customer segments for more effective advertising.

### A. Data Preparation
Three datasets were developed for analysis:
1. Transformation of the original dataset for suitability with predictive models.
2. Similar to the previous one with an added One Hot Encoding of the cluster labels, with the removal of one category to avert the dummy variable trap. This produced two feature groups: Group 1 and Group 2. 
3. Division of the dataset into three segments (tables) , each aligning with a cluster: Group 1, Group 2, and Group 3.

### B. Addressing Imbalance with SMOTE
To counter the imbalance in datasets (predominantly '0' responses), SMOTE was used to balance the classes, enhancing prediction reliability.

Dataset 1 and Dataset 2 contains 
- **Number of 0's:** 1906
- **Number of 1's:** 334

Dataset 3 (3 tables)
1. Group 0 contains
   - **Number of 0's:** 939
   - **Number of 1's:** 89
2. Group 1 contains
   - **Number of 0's:** 463
   - **Number of 1's:** 162
4. Group 2 contains
   - **Number of 0's:** 504
   - **Number of 1's:** 83
     
### C. Model Selection and Evaluation
XGBoost was chosen particularly for its efficiency in situations requiring minimal hyperparameter tuning. This aspect of XGBoost is crucial in this context, where the goal is to ascertain the impact of clustering on the predictive model's performance in response to an unseen feature like the "Latest Campaign". XGBoost's ability to deliver high performance with less need for hyperparameter optimization makes it suitable for quickly establishing a reliable baseline model. In our case, this means being able to accurately assess the effectiveness of customer segmentation (clustering) without the extensive calibration often required by other algorithms.

### D. Results

![Metrics_XGBOOST](https://github.com/eliasright/Customer/assets/151723828/de8d2b84-2c6d-45a3-b105-9cadca46b55d)

Analysis of the model's evaluation metrics showed:
- Improved performance metrics in Dataset 2 following the addition of One Hot Encoded cluster labels.
- Significant enhancement in all metrics for Group 0 in Dataset 3, suggesting high campaign receptiveness.
- Mixed results for Group 1 with varied performance across accuracy, precision, recall, and F1 score.
- Group 2's performance aligned with baseline expectations.

The metrics suggest that while overall accuracy might not always improve, the model's ability to correctly identify potential responders (as evidenced by recall and F1 score improvements) is enhanced, which is arguably more important in identifying target audience for a campaign.
