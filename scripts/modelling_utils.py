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

# Sys.
import sys

# Utils.
from scripts.eda_utils import check_outliers
from scripts.exception import CustomException
from datetime import datetime


def feature_engineering(df):
    '''
    This function performs feature engineering on a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing customer data.

    Returns:
        pd.DataFrame: A DataFrame with engineered features.

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
        # Renaming columns for better data manipulation and exploration.
        df.columns = [x.lower() for x in df.columns]

        # Converting the dt_customer column to datetime for feature engineering.
        df['dt_customer'] = pd.to_datetime(df['dt_customer'], format='%d-%m-%Y')

        # Dropping outlier rows representing inconsistent information.
        numerical_features = df.select_dtypes('number').columns.to_list()
        outlier_indexes, _, _ = check_outliers(data=df, features=numerical_features)
        to_drop_indexes = outlier_indexes['income'] + outlier_indexes['year_birth']
        df.drop(to_drop_indexes, inplace=True)

        # Feature engineering.
        
        # Merging some education and marital status categories.
        df['education'] = df['education'].map({'Graduation': 'Graduate', 'PhD': 'Postgraduate', 'Master': 'Postgraduate', '2n Cycle': 'Undergraduate', 'Basic': 'Undergraduate'})
        df['marital_status'] = df['marital_status'].map({'Single': 'Single', 'Divorced': 'Single', 'Widow': 'Single', 'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single', 'Married': 'Partner', 'Together': 'Partner'})

        # Creating a feature indicating the total number of campaigns accepted.
        df['total_accepted_cmp'] = df['acceptedcmp1'] + df['acceptedcmp2'] + df['acceptedcmp3'] + df['acceptedcmp4'] + df['acceptedcmp5']

        # Creating a feature indicating the total number of children at home, whatever they are teenhome or kidhome.
        df['children'] = df['kidhome'] + df['teenhome']

        # Creating a feature indicating customer's age.
        df['age'] = 2023 - df['year_birth']

        # Creating RFM model's features.

        # Creating a feature indicating the total number of purchases made by a customer to get the frequency.
        df['total_purchases'] = df['numcatalogpurchases'] + df['numdealspurchases'] + df['numstorepurchases'] + df['numwebpurchases']

        # Creating a relationship duration (in years, because of the long relationships present in the data) feature to get the frequency.
        current_date = datetime.today()
        df['relationship_duration'] = (current_date.year - df['dt_customer'].dt.year) 

        # Creating the frequency variable.
        df['frequency'] = df['total_purchases'] / df['relationship_duration']
        
        # Creating a monetary feature, indicating the total amount spent on company's products by the customer.
        df['monetary'] = df['mntfishproducts'] + df['mntfruits'] + df['mntgoldprods'] + df['mntmeatproducts'] + df['mntsweetproducts'] + df['mntwines']

        # Creating a feature indicating the average purchase value.
        df['avg_purchase_value'] = df['monetary'] / df['total_purchases'].replace(0, np.nan)

        # Dropping irrelevant columns.
        df.drop(columns=['z_costcontact', 'z_revenue', 'id', 'kidhome', 
                         'teenhome', 'complain', 'response', 
                        'acceptedcmp1', 'acceptedcmp2', 'acceptedcmp3', 
                        'acceptedcmp4', 'acceptedcmp5', 'dt_customer',
                        'year_birth', 'total_purchases'], inplace=True)

        # Dropping the inconsistent outlier from feature engineering.
        df.drop(df.loc[df['avg_purchase_value'] > 1500].index, inplace=True)

        return df

    except Exception as e:
        raise CustomException(e, sys)