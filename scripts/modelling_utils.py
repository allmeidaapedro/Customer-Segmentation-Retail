'''
This script aims to provide functions that will turn the modelling process easier
'''

'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modelling.
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, fcluster

# Sys.
import sys

# Utils.
from scripts.eda_utils import check_outliers
from scripts.exception import CustomException
from datetime import datetime


def feature_engineering(data):
    '''
    This function performs feature engineering on a given DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing customer data.

    Returns:
        feat_eng_df: A DataFrame with engineered features.
        ids: A pandas series containing the final customers' ID's.

    Raises:
        CustomException: An exception is raised if any errors occur during the process.

    Description:
        - Renames columns to lowercase for better data manipulation.
        - Converts the 'dt_customer' column to datetime format.
        - Removes inconsistent outlier rows based on income and year of birth.
        - Merges some education and marital status categories.
        - Creates features for the total number of accepted campaigns, total number of children at home, customer's age, RFM model features, frequency, monetary value, and average purchase value.
        - Drops irrelevant columns.
        - Removes inconsistent outliers based on average purchase value.

    Example:
        df = feature_engineering(df)
    '''

    try:
        # Getting a copy to avoid modifying the data.
        feat_eng_df = data.copy()

        # Renaming columns for better data manipulation and exploration.
        feat_eng_df.columns = [x.lower() for x in feat_eng_df.columns]

        # Converting the dt_customer column to datetime for feature engineering.
        feat_eng_df['dt_customer'] = pd.to_datetime(feat_eng_df['dt_customer'], format='%d-%m-%Y')

        # Dropping outlier rows representing inconsistent information.
        numerical_features = feat_eng_df.select_dtypes('number').columns.to_list()
        outlier_indexes, _, _ = check_outliers(data=feat_eng_df, features=numerical_features, verbose=False)
        to_drop_indexes = outlier_indexes['income'] + outlier_indexes['year_birth']
        feat_eng_df.drop(to_drop_indexes, inplace=True)

        # Feature engineering.
        
        # Merging some education and marital status categories.
        feat_eng_df['education'] = feat_eng_df['education'].map({'Graduation': 'Graduate', 'PhD': 'Postgraduate', 'Master': 'Postgraduate', '2n Cycle': 'Undergraduate', 'Basic': 'Undergraduate'})
        feat_eng_df['marital_status'] = feat_eng_df['marital_status'].map({'Single': 'Single', 'Divorced': 'Single', 'Widow': 'Single', 'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single', 'Married': 'Partner', 'Together': 'Partner'})

        # Creating a feature indicating the total number of campaigns accepted.
        feat_eng_df['total_accepted_cmp'] = feat_eng_df['acceptedcmp1'] + feat_eng_df['acceptedcmp2'] + feat_eng_df['acceptedcmp3'] + feat_eng_df['acceptedcmp4'] + feat_eng_df['acceptedcmp5']

        # Creating a feature indicating the total number of children at home, whatever they are teenhome or kidhome.
        feat_eng_df['children'] = feat_eng_df['kidhome'] + feat_eng_df['teenhome']

        # Creating a feature indicating customer's age.
        feat_eng_df['age'] = 2023 - feat_eng_df['year_birth']

        # Creating RFM model's features.

        # Creating a feature indicating the total number of purchases made by a customer to get the frequency.
        feat_eng_df['total_purchases'] = feat_eng_df['numcatalogpurchases'] + feat_eng_df['numdealspurchases'] + feat_eng_df['numstorepurchases'] + feat_eng_df['numwebpurchases']

        # Creating a relationship duration (in years, because of the long relationships present in the data) feature to get the frequency.
        current_date = datetime.today()
        feat_eng_df['relationship_duration'] = (current_date.year - feat_eng_df['dt_customer'].dt.year) 

        # Creating the frequency variable.
        feat_eng_df['frequency'] = feat_eng_df['total_purchases'] / feat_eng_df['relationship_duration']
        
        # Creating a monetary feature, indicating the total amount spent on company's products by the customer.
        feat_eng_df['monetary'] = feat_eng_df['mntfishproducts'] + feat_eng_df['mntfruits'] + feat_eng_df['mntgoldprods'] + feat_eng_df['mntmeatproducts'] + feat_eng_df['mntsweetproducts'] + feat_eng_df['mntwines']

        # Creating a feature indicating the average purchase value.
        feat_eng_df['avg_purchase_value'] = feat_eng_df['monetary'] / feat_eng_df['total_purchases'].replace(0, np.nan)

        # Dropping the inconsistent outlier from feature engineering.
        feat_eng_df.drop(feat_eng_df.loc[feat_eng_df['avg_purchase_value'] > 1500].index, inplace=True)
        
        # Obtaining the ID column for further use in customer segmentation.
        ids = feat_eng_df['id']

        # Dropping irrelevant columns.
        feat_eng_df.drop(columns=['z_costcontact', 'z_revenue', 'id', 'kidhome', 
                         'teenhome', 'complain', 'response', 
                        'acceptedcmp1', 'acceptedcmp2', 'acceptedcmp3', 
                        'acceptedcmp4', 'acceptedcmp5', 'dt_customer',
                        'year_birth', 'total_purchases'], inplace=True)

        return feat_eng_df, ids

    except Exception as e:
        raise CustomException(e, sys)
    

def silhouette_analysis(data, model, k_list=np.arange(2, 10, 1)):
    '''
    Perform silhouette analysis for a clustering model.

    This function takes a dataset, a clustering model, and a range of cluster numbers to perform silhouette analysis.
    It plots the silhouette scores for different cluster numbers, as well as individual silhouette plots for each cluster
    configuration.

    Parameters:
        data (array-like): The input data for clustering.
        model: The clustering model to evaluate.
        k_list (array-like, optional): A range of cluster numbers to analyze. Default is np.arange(2, 10, 1).

    Returns:
        None

    Example:
        silhouette_analysis(data, KMeans(), k_list=[2, 3, 4, 5])

    Raises:
        CustomException: An exception that wraps other exceptions raised during the function execution.
    
    Note:
        This function uses Matplotlib for plotting and assumes that it has been properly imported.

    '''
    try:
        labels_at_k, silhouette_scores_at_k, silhouette_plots = [], [], []

        fig, ax = plt.subplots(int(np.ceil(len(k_list) / 2)), 2, figsize=(20, 20))
        fig.subplots_adjust(hspace=0.3)

        min_x_tick = 0

        for k in k_list:
            row, column = divmod(k, 2)

            # Set the x-axis and y-axis limits for each subplot
            ax[row - 1, column].set_xlim([-0.1, 1])
            ax[row - 1, column].set_ylim([0, len(data) + (k + 1) * 20])

            # Fitting the model and obtaining silhouette scores for k.
            model_name = type(model).__name__

            if model_name == 'KMeans':
                model = KMeans(init='k-means++', n_clusters=k, n_init=50, random_state=42)
                model.fit(data)
                labels = model.labels_

            elif model_name == 'GaussianMixture':
                model = GaussianMixture(n_components=k, n_init=25, random_state=42)
                model.fit(data)
                labels = model.predict(data)

            else:
                model = linkage(data, 'ward')
                labels = fcluster(model, k, criterion='maxclust')

            labels_at_k.append(k)
            ss_score = silhouette_score(data, labels=labels, metric='euclidean')
            silhouette_scores_at_k.append(ss_score)
            samples_silhouette_scores = silhouette_samples(data, labels)

            y_lower = 20

            for i in range(k):

                ith_samples_silhouette_scores = samples_silhouette_scores[labels == i]
                ith_samples_silhouette_scores.sort()
                ith_cluster_size = ith_samples_silhouette_scores.shape[0]
                y_upper = y_lower + ith_cluster_size

                cmap = plt.get_cmap('rainbow')
                color = cmap(i / k)

                # Plot the silhouette values for each cluster
                ax[row - 1, column].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_samples_silhouette_scores, color=color, alpha=1)
                ax[row - 1, column].vlines(ss_score, -10, data.shape[0], linestyle='--', color='black', linewidth=2, label='Average Silhouette Score' if i == 0 else '')
                ax[row - 1, column].set_title(f'Average Silhouette Score for {k} Clusters = {ss_score:.4f}')
                ax[row - 1, column].set_ylim([-10, len(data) + (k + 1) * 20])
            
                x_tick = np.round(min(samples_silhouette_scores), 1)

                y_lower = y_upper + 20

            if x_tick < min_x_tick:
                min_x_tick = x_tick

            # Store the silhouette plot for later use
            silhouette_plots.append(ax[row - 1, column])


        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(k_list, silhouette_scores_at_k, 'k--', marker='o')
        ax.vlines(k_list[np.argmax(silhouette_scores_at_k)], silhouette_scores_at_k[np.argmin(silhouette_scores_at_k)],
                silhouette_scores_at_k[np.argmax(silhouette_scores_at_k)], linestyle='--', color='orange', label='Max Silhouette')

        ax.set_ylabel('Silhouette Score')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_title(f'Silhouette Scores for {type(model).__name__}')
        ax.legend()

        # Plot the silhouette plots with adjusted ticks and labels
        for silhouette_plot in silhouette_plots:
            plt.sca(silhouette_plot)
            plt.yticks(np.arange(0, len(data), int(len(data)/5)))

            plt.xlim([min_x_tick, 1])
            plt.xticks(np.arange(x_tick, 1.05, 0.1))

        plt.show()

    except Exception as e:
        raise CustomException(e, sys)