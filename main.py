import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import time
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from kneed import KneeLocator
import warnings
from sklearn.utils import resample
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import warnings
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import xgboost as xgb

#####################################
################ MAIN ###############
#####################################

def main():
    """
    Main function to execute the customer data analysis pipeline.

    This function includes steps for data reading, preprocessing, clustering,
    customer profiling, and prediction using machine learning models.
    """

    # Read customer data from a CSV file
    df = pd.read_csv("customer_data.csv", delimiter="\t")
    
    # Rename columns for better readability
    rename(df)

    # Optional: Uncomment to view an overview of the customer data
    data_overview(df)

    # Transform the user data for clustering (Data prep)
    # 'df_Transform': Data transformed for clustering (campaign data dropped and standardized)
    # 'df_New': Data transformed without dropping campaign data and without standardization
    # 'df_Predict': Data transformed for prediction (includes campaign data and standardized)
    df_Transform = transform_customer_data(df, dropCampaign=True, standardizeIt=True, printIt=False)
    df_New = transform_customer_data(df, dropCampaign=False, standardizeIt=False, printIt=False)
    df_Predict = transform_customer_data(df, dropCampaign=False, standardizeIt=True, printIt=False)

    # Perform customer segmentation using clustering
    labels = customer_segmentation(df_Transform=df_Transform, graphIt=True)

    # Profile customers based on clustering results
    customer_profile(df=df_New, df_Transform=df_Transform, labels=labels)

    # Predict customer responses using machine learning models
    customer_prediction(df=df_Predict, labels=labels)
    
#######################################
########## 1. DATA OVERVIEW ###########
#######################################

def data_overview(df):
    """
    This function provides an overview of a given DataFrame. It includes:
    - Dataset shape
    - Missing fields
    - Duplicate ID checks
    - Duplicate customer entries based on personal details
    - Outliers in personal information
    - Value counts for categorical columns
    - Descriptive statistics for purchases and purchasing methods
    - A heatmap of Pearson correlations for purchase categories
    """

    # Display shape of the dataset
    print("########## Shape of Dataset ##########")
    print(df.shape, "\n")

    # Display missing fields in the dataset
    print("########## Missing fields ##########")
    print(df.isnull().sum(), "\n")

    # Check and display the number of duplicate IDs
    duplicates_id = df['ID'].duplicated().sum()
    print("########## Duplicate IDs ##########")
    print(f"Number of duplicate IDs: {duplicates_id}\n")

    # Find and display details about duplicated customer entries
    results = count_duplicates(df)
    print("########## Column Difference Counts in Duplicate Groups ##########")
    for col, count in results['column_differences'].items():
        if count > 0:
            print(f"Column {col}: {count} differences")
    print(f"Number of pure duplication groups: {results['pure_duplication_count']} \n")

    # Check for outliers in personal information
    outlier_columns = ['Income', 'Kid', 'Teen']
    outlier_data = df[outlier_columns].copy()
    outlier_data['Age'] = 2014 - df['Year_Birth']
    print("########## Outlier Graph Produced ########## \n")
    create_outlier_boxplots(outlier_data)

    # Display outliers for age
    print("########## Outlier for Age ##########")
    print(df[df['Year_Birth'] < 1930].assign(Age=2014 - df['Year_Birth'])[['Year_Birth', 'Age', 'Income', 'Kid', 'Teen']], '\n')

    # Display frequency of different education levels
    education_values = df['Education'].value_counts()
    print("########## Education Values and Frequencies ##########")
    print(education_values, "\n")

    # Display frequency of different marital statuses
    marital_status_values = df['Marital_Status'].value_counts()
    print("########## Marital Status Values and Frequencies ##########")
    print(marital_status_values, "\n")

    # Display frequency of complaint values
    complain_values = df['Complain'].value_counts()
    print("########## Complain Values and Frequencies ##########")
    print(complain_values, "\n")

    # Display descriptive statistics for purchases
    describe_subset_2 = df[['Wine', 'Fruit', 'Meat', 'Fish', 'Sweet', 'Gold']].describe().T
    print("########## Description of Purchases ##########")
    print(describe_subset_2, "\n")

    # Display a heatmap of Pearson correlations for purchase categories
    correlation_matrix = df[['Wine', 'Fruit', 'Meat', 'Fish', 'Sweet', 'Gold']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.xticks(rotation=0)
    plt.show()

    # Display descriptive statistics for purchasing methods
    describe_subset_3 = df[['Web Purchases', 'Catalog Purchases', 'Store Purchases', 'Web Visits Month']].describe().T
    print("########## Description of Purchasing Methods ##########")
    print(describe_subset_3, "\n")

def count_duplicates(df):
    """
    This function identifies duplicate customer entries in the DataFrame.
    It focuses on personal details like year of birth, education, marital status, income, etc., and 
    checks for duplicates based on these details. It also counts differences in other columns for these duplicate groups.

    Args:
    df (DataFrame): The DataFrame to analyze for duplicates.

    Returns:
    dict: A dictionary containing counts of column differences in duplicate groups and the count of pure duplication groups.
    """

    # Define customer detail columns to check for duplicates
    customer_details = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kid', 'Teen']

    # Identify duplicates based on customer details
    promotional_duplicates = df[df.duplicated(subset=customer_details, keep=False)]

    # Define columns to check for differences in duplicate groups
    column_diff_columns = df.columns.difference(customer_details + ['ID'])
    column_diff_counts = pd.Series(0, index=column_diff_columns)

    def count_column_differences(group, column_diff_counts):
        """
        Counts the differences in columns for each group of duplicates.

        Args:
        group (DataFrame): Group of duplicated rows.
        column_diff_counts (Series): Series to keep track of column differences.

        Returns:
        bool: True if the group is a pure duplication, False otherwise.
        """
        first_row = group.iloc[0]
        pure_duplication = True
        for _, row in group.iterrows():
            for col in column_diff_columns:
                if row[col] != first_row[col]:
                    column_diff_counts[col] += 1
                    pure_duplication = False
                    break
        return pure_duplication

    # Count pure duplication groups
    pure_duplication_count = 0
    for _, group in promotional_duplicates.groupby(customer_details):
        if count_column_differences(group, column_diff_counts):
            pure_duplication_count += 1

    return {'column_differences': column_diff_counts, 'pure_duplication_count': pure_duplication_count}

def create_outlier_boxplots(outlier_data):
    """
    Creates boxplots for each column in the outlier_data DataFrame to visualize outliers.

    Args:
    outlier_data (DataFrame): DataFrame containing the data for which outliers are to be visualized.

    This function sets up a seaborn style, calculates the layout for the plots based on the number of columns,
    and creates box plots for each column with annotated whisker values.
    """

    # Set the style and color palette for seaborn plots
    sns.set(style="whitegrid", palette="pastel")

    # Determine the layout for the subplots based on the number of columns
    num_columns = len(outlier_data.columns)
    num_rows = np.ceil(num_columns / 2).astype(int)
    num_cols_per_row = min(num_columns, 2)

    # Initialize the figure with calculated dimensions
    fig_width = 8 * num_cols_per_row
    fig_height = 4 * num_rows
    fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=(fig_width, fig_height))

    # Flatten axes array if there are more than one row of plots
    if num_rows > 1:
        axes = axes.flatten()

    # Main title for all plots
    main_title = "Distribution of Outlier Columns with Outliers"

    # Create and annotate box plots for each column
    for i, col in enumerate(outlier_data.columns):
        ax = axes[i] if num_rows > 1 else axes
        sns.boxplot(y=outlier_data[col], ax=ax, color="lightblue")
        ax.set_title(f'{col} Distribution', fontsize=12)
        ax.set_ylabel(col, fontsize=10)
        ax.set_xlabel('')

        # Calculate and annotate whisker values for each box plot
        q1 = outlier_data[col].quantile(0.25)
        q3 = outlier_data[col].quantile(0.75)
        iqr = q3 - q1
        upper_whisker = outlier_data[col][outlier_data[col] <= q3 + 1.5 * iqr].max()
        lower_whisker = outlier_data[col][outlier_data[col] >= q1 - 1.5 * iqr].min()
        ax.text(0.95, upper_whisker, f'{upper_whisker:.2f}', verticalalignment='center', horizontalalignment='right', transform=ax.get_yaxis_transform(), fontsize=10, color='red')
        ax.text(0.95, lower_whisker, f'{lower_whisker:.2f}', verticalalignment='center', horizontalalignment='right', transform=ax.get_yaxis_transform(), fontsize=10, color='red')

        ax.tick_params(labelsize=10)

    # Hide unused axes if there are any
    if num_rows * num_cols_per_row > num_columns:
        for j in range(num_columns, num_rows * num_cols_per_row):
            fig.delaxes(axes[j])

    # Adjust layout and add main title
    plt.subplots_adjust(wspace=3, hspace=3)
    plt.suptitle(main_title, fontsize=15, y=0.98)
    plt.tight_layout()
    plt.show()

def rename(df):
    """
    Renames columns in the DataFrame for better readability.

    Args:
    df (DataFrame): The DataFrame whose columns are to be renamed.

    This function renames several columns in the DataFrame to more descriptive names, 
    aiding in the clarity and understanding of the data.
    """

    # Renaming columns for better readability and understanding
    df.rename(columns={
        'Kidhome': 'Kid',
        'Teenhome': 'Teen',
        'MntWines': 'Wine',
        'MntFruits': 'Fruit',
        'MntMeatProducts': 'Meat',
        'MntFishProducts': 'Fish',
        'MntSweetProducts': 'Sweet',
        'MntGoldProds': 'Gold',
        'NumDealsPurchases': 'Deals Purchases',
        'NumWebPurchases': 'Web Purchases',
        'NumCatalogPurchases': 'Catalog Purchases',
        'NumStorePurchases': 'Store Purchases',
        'NumWebVisitsMonth': 'Web Visits Month',
        'AcceptedCmp1': 'Campaign 1',
        'AcceptedCmp2': 'Campaign 2',
        'AcceptedCmp3': 'Campaign 3',
        'AcceptedCmp4': 'Campaign 4',
        'AcceptedCmp5': 'Campaign 5',
        'Response' : 'Latest Campaign'
    }, inplace=True)

#######################################
##### 2. TRANSFORM CUSTOMER DATA ######
#######################################

def transform_customer_data(df_original, dropCampaign=False, standardizeIt=False, printIt=False):
    """
    Transforms customer data for analysis by adding new features, encoding categorical variables, 
    handling outliers, and optionally standardizing the data.

    Args:
    df_original (DataFrame): The original DataFrame to transform.
    dropCampaign (bool): If True, campaign columns are dropped from the DataFrame.
    standardizeIt (bool): If True, standardizes the numerical columns in the DataFrame.
    printIt (bool): If True, prints the DataFrame before and after transformation.

    Returns:
    DataFrame: The transformed DataFrame.
    """

    # Create a copy of the original DataFrame to avoid modifying it
    df = df_original.copy()

    # Convert 'Dt_Customer' to datetime and compute membership length
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
    max_new_date = (df['Dt_Customer'] + pd.to_timedelta(df['Recency'], unit='D')).max()
    df["Membership Length"] = (max_new_date - df['Dt_Customer']).dt.days

    # Add a new 'Age' column
    df["Age"] = max_new_date.year - df["Year_Birth"]

    # One-hot encoding for 'Education' and 'Marital_Status'
    df = pd.concat([df, pd.get_dummies(df['Education'], prefix='Edu', prefix_sep=' ')], axis=1)
    df['Marital_Status'] = df['Marital_Status'].replace(['Alone', 'Absurd', 'YOLO'], 'Single')
    df = pd.concat([df, pd.get_dummies(df['Marital_Status'], prefix='Marital', prefix_sep=' ')], axis=1)

    # Drop original columns replaced by new features or encoded
    df.drop(columns=['ID', 'Marital_Status', 'Education', 'Dt_Customer', 'Year_Birth'], inplace=True)

    # Option to drop campaign-related columns
    if dropCampaign:
        df.drop(columns=['Campaign 1', 'Campaign 2', 'Campaign 3', 'Campaign 4', 'Campaign 5', 'Latest Campaign'], inplace=True)

    # Drop columns with unclear purpose
    df.drop(columns=['Z_CostContact', 'Z_Revenue'], inplace=True)

    # Handle outliers in 'Age' and 'Income'
    df.loc[df['Age'] > 100, 'Age'] = np.nan
    df.loc[df['Income'] > 500000, 'Income'] = np.nan

    # Select columns for standardization
    columns_to_standardize = [col for col in df.columns if df[col].nunique() > 3 and df[col].dtype != 'bool' and col not in ['Income', 'Age']]

    # Create a copy for standardization
    df_standardized = df.copy()

    # Apply standardization
    scalers = {}
    for col in columns_to_standardize:
        scalers[col] = StandardScaler()
        df_standardized[col] = scalers[col].fit_transform(df_standardized[[col]])

    # Impute missing values for 'Income' and 'Age'
    imputer = KNNImputer(n_neighbors=5)
    df_standardized[['Income', 'Age']] = imputer.fit_transform(df_standardized[['Income', 'Age']])
    df[['Income', 'Age']] = df_standardized[['Income', 'Age']]

    # Print data before and after transformation, if required
    if printIt:
        print("########## Description of Purchases Before Transformation ##########")
        print(df_original.head(), "\n")
        print("########## Description of Purchases After Transformation ##########")
        print(df.head(), "\n")

    # Additional option to standardize all relevant columns
    if standardizeIt:
        scaler = StandardScaler()
        columns_to_standardize = [col for col in df.columns if df[col].nunique() > 3 and df[col].dtype != 'bool']
        df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])

    return df


def remove_duplicates(input_df):
    """
    Removes duplicate rows from a DataFrame based on specific customer details.

    Args:
    input_df (DataFrame): The DataFrame from which duplicates will be removed.

    Returns:
    DataFrame: The DataFrame with duplicates removed.
    """

    # Create a copy of the input DataFrame to avoid modifying the original data
    df = input_df.copy()

    # Define the columns based on which duplicates will be identified
    customer_details = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome']

    # Identify and print the count of duplicated rows
    duplicates = df[df.duplicated(subset=customer_details, keep=False)]
    print("Number of duplicated rows: ", len(duplicates))

    # Function to select a row to keep from each group of duplicates
    def select_row_to_keep(group):
        if group['Latest Campaign'].max() == 1:
            return group[group['Latest Campaign'] == 1].head(1)
        return group.head(1)

    # Apply the function to select rows to keep
    selected_rows = duplicates.groupby(customer_details, group_keys=False).apply(select_row_to_keep)

    # Identify and remove rows that were not selected
    rows_to_drop = duplicates.drop(selected_rows.index)
    print("Removed number of duplicated rows: ", len(rows_to_drop))
    df = df.drop(rows_to_drop.index)

    return df

#######################################
###### 3. CUSTOMER SEGMENTATION #######
#######################################

def customer_segmentation(df_Transform, graphIt=False):
    """
    Performs customer segmentation using KMeans clustering. 
    It utilizes an elbow test to find the optimal number of clusters.

    Args:
    df_Transform (DataFrame): The DataFrame to be used for segmentation.
    graphIt (bool): If True, the elbow test graph will be displayed.

    Returns:
    array: Array of cluster labels for each data point.
    """
    # Find the optimal number of clusters using the elbow test
    n_clusters = elbow_test(df_Transform, 1, 10, graphIt)

    # Apply KMeans clustering with the determined number of clusters
    K_Labels = KMeans_cluster(df=df_Transform, n_clusters=n_clusters)

    return K_Labels

def elbow_test(df, start_k, end_k, graphIt=False):
    """
    Performs an elbow test to determine the optimal number of clusters for KMeans.
    The function calculates distortions for a range of cluster numbers and identifies
    the 'elbow point' where the rate of decrease in distortion diminishes.

    Args:
    df (DataFrame): The DataFrame on which clustering is to be performed.
    start_k (int): The starting number of clusters for testing.
    end_k (int): The ending number of clusters for testing.
    graphIt (bool): If True, the elbow graph will be displayed.

    Returns:
    int: Optimal number of clusters determined by the elbow test.
    """
    # Initialize list to store distortions for different cluster counts
    distortions = []
    K = range(start_k, end_k + 1)

    # Calculate distortions for each cluster count
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)

    # Determine the 'elbow point' using KneeLocator
    kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
    knee_point = kn.knee

    # Optionally graph the elbow plot
    if graphIt:
        # Plotting Elbow Method graph with distortion and rate of change
        plot_elbow_graph(K, distortions, knee_point)

    return knee_point

def plot_elbow_graph(K, distortions, knee_point):
    """
    Plots the elbow graph showing distortions for each cluster count and the calculated knee point.

    Args:
    K (range): Range of cluster counts tested.
    distortions (list): List of distortions for each cluster count.
    knee_point (int): Calculated knee point indicating optimal cluster count.
    """
    # Plotting setup
    fig, ax1 = plt.subplots(figsize=(16, 8))
    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Distortion', color=color)
    ax1.plot(K, distortions, 'bx-', markeredgewidth=2, markersize=5, label='Distortion')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(K)  # Set x-axis ticks

    # Secondary axis for rate of change
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Rate of Change of Distortion', color=color)
    ax2.plot(K[1:], np.diff(distortions), 'ro-', markeredgewidth=2, markersize=5, label='Derivative')
    ax2.tick_params(axis='y', labelcolor=color)

    # Highlighting and annotating the elbow point
    plt.axvline(x=knee_point, color='red', linestyle='--', linewidth=2)
    plt.text(knee_point, max(distortions), f'Optimal N: {knee_point}', horizontalalignment='right', color='red', fontsize=12)
    plt.title('Elbow Method For Optimal N with Rate of Change')
    plt.show()

def calculate_pca_variance(df):
    """
    Calculates the number of principal components needed to explain at least 70% of the variance.

    Args:
    df (DataFrame): The DataFrame for which PCA is to be calculated.

    Returns:
    int: The number of principal components that explain at least 70% of the variance.
    """
    # Initialize PCA and fit to the DataFrame
    pca = PCA(random_state=1221)
    pca.fit(df)

    # Calculate the cumulative explained variance ratio
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Determine the number of components explaining at least 70% variance
    target_variance = 0.70
    pca_70_percent_index = np.where(cumulative_variance >= target_variance)[0][0]

    return pca_70_percent_index + 1  # Adjust for zero-based index

def KMeans_cluster(df, n_clusters):
    """
    Applies Principal Component Analysis (PCA) and KMeans clustering to the DataFrame.

    Args:
    df (DataFrame): The DataFrame on which clustering is to be performed.
    n_clusters (int): The number of clusters for the KMeans algorithm.

    Returns:
    array: Array of cluster labels for each row in the DataFrame.
    """
    # Determine the optimal number of PCA components
    pca_var = calculate_pca_variance(df)

    # Apply PCA and transform the DataFrame
    pca = PCA(n_components=pca_var, random_state=1221)
    transformed_df = pca.fit_transform(df)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(transformed_df)
    labels = kmeans.labels_

    return labels

#######################################
######## 4. CUSTOMER PROFILE ##########
#######################################

def customer_profile(df, df_Transform, labels):
    """
    Creates a customer profile based on the given DataFrame and cluster labels. 
    It visualizes various aspects of customer data for each cluster.

    Args:
    df (DataFrame): Original DataFrame with customer data.
    df_Transform (DataFrame): Transformed DataFrame used for clustering.
    labels (array): Cluster labels for each data point in df_Transform.

    The function generates visualizations for income, spending, age, education,
    marital status, parental status, and response to campaigns for different clusters.
    """

    # Visualize clusters in 3D space
    cluster_plot(df_Transform, labels)

    # Compare income and spending across clusters
    income_spending(df, labels)
    box_whisker_plot(df, labels, 'Income')
    box_whisker_plot(df, labels, 'Spending')

    # Compare age distribution across clusters
    box_whisker_plot(df, labels, 'Age')

    # Compare education levels across clusters
    education_categories = ['Edu 2n Cycle', 'Edu Basic', 'Edu Graduation', 'Edu Master', 'Edu PhD']
    plot_categorical(df, education_categories, labels, 'Education Distribution')

    # Compare marital status across clusters
    marital_categories = ['Marital Divorced', 'Marital Married', 'Marital Single', 'Marital Together', 'Marital Widow']
    plot_categorical(df, marital_categories, labels, 'Marital Status Distribution')

    # Visualize distribution of parents (with kids and teens) across clusters
    plot_kid_teen_distribution(df, labels)

    # Compare responses to marketing campaigns across clusters
    campaign_categories = ['Campaign 1', 'Campaign 2', 'Campaign 3', 'Campaign 4', 'Campaign 5', 'Latest Campaign']
    df['Non Responsive'] = df[campaign_categories].apply(lambda row: all(row == 0), axis=1)
    campaign_categories.append('Non Responsive')
    plot_categorical(df, campaign_categories, labels, 'Campaign Distribution')

def plot_categorical(df, categories, labels, title):
    """
    Creates bar plots for categorical data, showing counts and proportions for each category across clusters.

    Args:
    df (DataFrame): DataFrame containing the data to plot.
    categories (list): List of categories to plot.
    labels (array): Cluster labels for each data point in the DataFrame.
    title (str): Title for the plots.

    The function generates two bar plots for each category: one showing the count and another showing the proportion.
    """

    # Add cluster labels to the DataFrame
    df_with_labels = df.assign(Cluster=labels)

    # Group data by clusters and calculate counts and proportions for each category
    grouped_df = df_with_labels.groupby('Cluster')[categories]
    category_counts_by_cluster = grouped_df.sum()
    category_proportions_by_cluster = (grouped_df.sum().T / grouped_df.sum().sum(axis=1)).T

    # Create subplot figures for counts and proportions
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot counts for each category by cluster
    category_counts_by_cluster.plot(kind='bar', ax=axes[0], legend=False)
    axes[0].set_title(f'{title} - Count')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Count')

    # Plot proportions for each category by cluster
    category_proportions_by_cluster.plot(kind='bar', stacked=True, ax=axes[1], legend=False)
    axes[1].set_title(f'{title} - Proportion')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Proportion')

    # Combine legends from both plots and place at the bottom of the figure
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(categories))

    # Adjust layout to fit legend and plots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()

def plot_kid_teen_distribution(df, labels):
    """
    Plots the distribution of households with kids, teens, both, or neither, categorized by customer clusters.

    Args:
    df (DataFrame): DataFrame containing the 'Kid' and 'Teen' columns.
    labels (array): Array of cluster labels.

    This function converts the 'Kid' and 'Teen' columns to binary, creates categories for household types,
    and plots the count and proportion of each type across different clusters.
    """

    # Convert 'Kid' and 'Teen' counts to binary (1 if count > 0, else 0)
    df['Kid'] = df['Kid'].apply(lambda x: 1 if x > 0 else 0)
    df['Teen'] = df['Teen'].apply(lambda x: 1 if x > 0 else 0)

    # Create a DataFrame for Kid, Teen, Both, Neither categories
    kid_teen_df = pd.DataFrame(index=df.index)
    kid_teen_df['Kid Only'] = ((df['Kid'] == 1) & (df['Teen'] == 0)).astype(int)
    kid_teen_df['Teen Only'] = ((df['Kid'] == 0) & (df['Teen'] == 1)).astype(int)
    kid_teen_df['Both'] = ((df['Kid'] == 1) & (df['Teen'] == 1)).astype(int)
    kid_teen_df['Neither'] = ((df['Kid'] == 0) & (df['Teen'] == 0)).astype(int)
    kid_teen_df['Cluster'] = labels

    # Group data by clusters and calculate counts and proportions
    category_counts_by_cluster = kid_teen_df.groupby('Cluster').sum()
    category_proportions_by_cluster = (category_counts_by_cluster.T / category_counts_by_cluster.sum(axis=1)).T

    # Plotting setup for counts and proportions
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    category_counts_by_cluster.plot(kind='bar', ax=axes[0], legend=False)
    axes[0].set_title('Kid and Teen Count Distribution by Cluster')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Count')
    category_proportions_by_cluster.plot(kind='bar', stacked=True, ax=axes[1], legend=False)
    axes[1].set_title('Kid and Teen Proportion Distribution by Cluster')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Proportion')

    # Combine legends and place at the bottom of the figure
    handles, labels = [], []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(labels))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def box_whisker_plot(df, labels, column):
    """
    Creates a box-and-whisker plot for a specified column in the DataFrame, categorized by clusters.

    Args:
    df (DataFrame): DataFrame containing the data.
    labels (array): Array of cluster labels.
    column (str): The name of the column to plot.

    If the column is 'Spending', it calculates the total spending from related columns.
    The function adds annotations for median, upper, and lower quartiles on the plot.
    """

    # Set seaborn style for the plot
    sns.set(style="whitegrid")

    # Prepare DataFrame with cluster labels
    clustered_df = df.assign(Cluster=labels)

    # Calculate 'Spending' if the column is specified
    if column == 'Spending':
        clustered_df['Spending'] = clustered_df[['Wine', 'Fruit', 'Meat', 'Fish', 'Sweet', 'Gold']].sum(axis=1)

    # Create boxplot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x='Cluster', y=column, data=clustered_df)
    plt.title(f'{column} Distribution by Cluster', fontsize=14)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel(column, fontsize=12)

    # Move the annotation on the X axis to avoid overlapping
    offset = 0.16

    # Annotate quartiles and median for each cluster
    num_clusters = len(clustered_df['Cluster'].unique())
    for i in range(num_clusters):
        cluster_data = clustered_df[clustered_df['Cluster'] == i][column]
        median = np.median(cluster_data)
        Q1 = np.percentile(cluster_data, 25)
        Q3 = np.percentile(cluster_data, 75)
        ax.text(i + offset, Q1, f'{Q1:.2f}', horizontalalignment='left', color='orange', weight='semibold')
        ax.text(i, median, f'{median:.2f}', horizontalalignment='center', color='yellow', weight='semibold')
        ax.text(i - offset, Q3, f'{Q3:.2f}', horizontalalignment='right', color='red', weight='semibold')

    plt.show()

def income_spending(df_input, labels):
    """
    Creates a scatter plot comparing income and total spending of customers, color-coded by cluster labels.

    Parameters:
    df_input (DataFrame): The DataFrame containing income and spending data.
    labels (array-like): Clustering labels for each data point in the DataFrame.

    The function calculates the total spending by summing specified columns and plots income against spending.
    """

    # Copy the DataFrame to avoid modifying the original data
    df = df_input.copy()

    # Calculate total spending by summing up relevant columns
    df['Spending'] = df[['Wine', 'Fruit', 'Meat', 'Fish', 'Sweet', 'Gold']].sum(axis=1)

    # Initialize the scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a scatter plot of Income vs Total Spending
    scatter = ax.scatter(df['Income'], df['Spending'], c=labels, cmap='viridis', s=15)

    # Set plot titles and labels
    ax.set_title('Income vs Total Spend')
    ax.set_xlabel('Income')
    ax.set_ylabel('Total Spending')

    # Add a legend to the plot
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    plt.tight_layout()
    plt.show()

def cluster_plot(df_input, labels):
    """
    Creates a 3D scatter plot of customer data after applying PCA, color-coded by cluster labels.

    Parameters:
    df_input (DataFrame): The DataFrame to be visualized.
    labels (array-like): Clustering labels for each data point in the DataFrame.

    The function applies PCA to reduce the data to three dimensions and then creates a 3D scatter plot.
    """

    # Create a copy of the input DataFrame
    df = df_input.copy()

    # Apply PCA to reduce the data to three dimensions
    pca = PCA(n_components=3)
    df_pca = pca.fit_transform(df)

    # Initialize the 3D scatter plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot with the PCA components
    scatter = ax.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], c=labels, cmap='viridis')

    # Set plot titles and axis labels
    ax.set_title('3D Cluster of Customer Segmentation')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    # Add a legend to the plot
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    plt.show()

#######################################
## 4. PREDICTION OF LATEST CAMPAIGN ###
#######################################

def customer_prediction(df, labels):
    """
    Performs customer prediction using XGBoost with different approaches: without label encoding,
    with label encoding, and for each label individually. Compares the model performance in each case.

    Args:
    df (DataFrame): The DataFrame containing customer data.
    labels (array-like): Cluster labels for each customer.

    The function evaluates three approaches of XGBoost modeling and plots their performance metrics.
    """

    # XGBoost without label encoding
    y_test, predictions_df = customer_prediction_XGBOOST(df)
    metrics_without_labels = evaluate_model(y_test, predictions_df)

    # XGBoost with label encoding
    one_hot = pd.get_dummies(labels, prefix='Cluster ', drop_first=True)
    df_with_label = pd.concat([df, one_hot], axis=1)
    y_test_with_label, predictions_df_with_label = customer_prediction_XGBOOST(df_with_label, target_column='Latest Campaign')
    metrics_with_labels = evaluate_model(y_test_with_label, predictions_df_with_label)

    # XGBoost for each label individually
    metrics_per_label = customer_prediction_per_label(df, labels)

    # Comparing the results
    plot_evaluation_metrics(metrics_without_labels, metrics_with_labels, metrics_per_label)

def customer_prediction_per_label(df, labels):
    """
    Performs customer prediction using XGBoost for each label individually and evaluates the model performance.

    Args:
    df (DataFrame): The DataFrame containing customer data.
    labels (array-like): Cluster labels for each customer.

    Returns:
    dict: A dictionary containing model performance metrics for each label.
    """

    # Initialize a dictionary to store metrics for each label
    metrics_per_label = {}

    # Convert labels to a pandas Series and get unique labels
    labels_series = pd.Series(labels)
    unique_labels = labels_series.unique()

    # Perform prediction and evaluate model for each label
    for label in unique_labels:
        df_subset = df[labels_series == label]
        y_test, predictions = customer_prediction_XGBOOST(df_subset, target_column='Latest Campaign', useSmote=True)
        metrics = evaluate_model(y_test, predictions)
        metrics_per_label[label] = metrics

    return metrics_per_label

def customer_prediction_XGBOOST(df, target_column='Latest Campaign', useSmote=False):
    """
    Performs customer prediction using an XGBoost classifier. 
    It allows the option to use SMOTE (Synthetic Minority Over-sampling Technique) for balancing the dataset.

    Args:
    df (DataFrame): The DataFrame containing the dataset for prediction.
    target_column (str, optional): The name of the target column. Defaults to 'Latest Campaign'.
    useSmote (bool, optional): Flag to indicate whether to use SMOTE for balancing the dataset. Defaults to False.

    Returns:
    tuple: A tuple containing the actual values of the test set (y_test) and the predictions made by the model.
    """

    # Splitting the data into features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    if useSmote:
        # Applying SMOTE for balancing the dataset
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
    else:
        # Combining features and target for undersampling process
        data = pd.concat([X, y], axis=1)

        # Separating the dataset into majority and minority classes
        majority_class = data[data[target_column] == 0]
        minority_class = data[data[target_column] == 1]

        # Performing undersampling on the majority class
        majority_class_undersampled = resample(majority_class, 
                                               replace=False, 
                                               n_samples=len(minority_class), 
                                               random_state=42)

        # Combining the undersampled majority class with the minority class
        data = pd.concat([majority_class_undersampled, minority_class])

        # Re-separating features and target
        X = data.drop(target_column, axis=1)
        y = data[target_column]

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1221)

    # Initializing the XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Training the model with the training data
    model.fit(X_train, y_train)

    # Making predictions on the test set
    predictions = model.predict(X_test)

    return y_test, predictions


def evaluate_model(y_true, y_pred):
    """
    Evaluates the performance of a classification model using various metrics.

    Args:
    y_true (array-like): True labels of the test dataset.
    y_pred (array-like): Predicted labels by the model.

    Returns:
    dict: A dictionary containing accuracy, recall, precision, F1 score, and a full classification report.
    """

    # Calculate basic evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Generate a detailed classification report
    report = classification_report(y_true, y_pred)

    # Return all metrics in a dictionary
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1_score": f1,
        "classification_report": report
    }

def plot_evaluation_metrics(metrics_without_labels, metrics_with_labels, metrics_per_label):
    """
    Plots evaluation metrics for models with and without label encoding and for each individual label.

    Args:
    metrics_without_labels (dict): Metrics for the model without label encoding.
    metrics_with_labels (dict): Metrics for the model with label encoding.
    metrics_per_label (dict): Metrics for the model for each label.

    This function creates a bar plot comparing the performance metrics of different models.
    """

    # Extract metrics names and values for plotting
    metric_names = list(metrics_without_labels.keys())[:-1]  # Exclude the classification report
    values_without_labels = [metrics_without_labels[name] for name in metric_names]
    values_with_labels = [metrics_with_labels[name] for name in metric_names]
    values_per_label = {cluster: [metrics[name] for name in metric_names] 
                        for cluster, metrics in metrics_per_label.items()}

    # Setting up the bar plot
    n_groups = len(metric_names)
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['purple', 'yellow', 'green']  # Custom colors for clusters

    # Helper function to plot bars
    def plot_bars(values, positions, label, color=None):
        bars = ax.bar(positions, values, width=bar_width, label=label, color=color)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize='x-small')

    # Plot bars for models without and with label encoding
    plot_bars(values_without_labels, [pos - bar_width / 2 for pos in range(n_groups)], 'Without Cluster')
    plot_bars(values_with_labels, [pos + bar_width / 2 for pos in range(n_groups)], 'With Cluster')

    # Plot bars for each cluster
    for i, (cluster, values) in enumerate(values_per_label.items()):
        plot_bars(values, [pos + bar_width * (i + 1.5) for pos in range(n_groups)], f'Cluster {cluster}', colors[i % len(colors)])

    # Set x-axis labels and configure plot layout
    plt.xticks([pos + bar_width * len(values_per_label) / 2 for pos in range(n_groups)], metric_names)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Model Evaluation Metrics Comparison')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(values_per_label) + 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#######################################
################ MAIN #################
#######################################

if __name__ == "__main__":
    # Filter out FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Run main
    main()