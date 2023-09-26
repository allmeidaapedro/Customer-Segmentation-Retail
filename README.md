# Retail Store Customer Segmentation and Profiling

<img src="reports/cseg.png" width=800px height=350px>

# 1. Description
- In this project, I performed an unsupervised learning clustering task using K-Means on unlabeled training data to segment and profile customers for a retail store. 
- After segmenting the clients, a loyalty program called "Prosperous" was developed based on the profile of our best customers, the Prosperous ones. 
- By utilizing techniques such as dimensionality reduction and silhouette score analysis for model comparison and cluster definition, I was able to effectively segment the clientele into five groups, creating distinct personas. 
- Finally, a financial estimate was made. The loyalty program has the potential to increase the total store revenue by 9%, amounting to $125,228.55. Therefore, the project is worthwhile.
- The project follows a real data science project workflow, encompassing tasks from data collection and exploratory data analysis (EDA) to final modeling. It includes features like exception handling, virtual environments, modular coding, code versioning (using Git and Github), and specifying project requirements. By structuring it this way, I've organized the entire project as a package, making it easily reproducible by others.

# 2. Technologies and tools
The technologies and tools used were Python (Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn), Jupyter Notebook, Git and Github (version control), machine learning clustering algorithms, statistics, Anaconda (terminal and virtual environment) and Visual Studio Code (project development environment).

# 3. Project Structure
Each folder/file content and purpose is described below:

Artifacts: Store machine learning model artifacts, including the raw data, preprocessed data, the preprocessor, and the model.

Notebooks:
- eda.ipynb: This notebook contains the exploratory data analysis.
- modelling.ipynb: This notebook covers the clustering machine learning modeling.
- data: This folder contains the raw data to be used in notebooks.

Reports: Store some images that are used for storytelling.

Scripts: Store utility scripts used for EDA, modeling, exception handling, and artifacts retrieval. The files names are self explainable.
- artifacts_utils.py
- eda_utils.py
- exception.py
- modelling_utils.py

Requirements, setup, gitignore, readme: The file setup.py allows me to build my entire project as a package, containing metadata and so on. Moreover, requirements.txt list all the dependencies needed for the project with the specific versions for reproducibility. Gitignore allows me to hide irrelevant information from commits and I am using readme.md for documentation and storytelling.


# 4. Business problem and project objectives

Problem statement:
- A retail store aims to gain a deeper understanding of its customer characteristics and to strategically utilize this knowledge. One of its primary goals is to proficiently segment its clientele, allowing for enhanced comprehension of its customers' preferences. This, in turn, facilitates the adaptation of products to cater to the unique requirements, behaviors, and concerns of various customer segments.
- Additionally, the store wants to build better connections with customers and keep them coming back by starting a loyalty program. They'll look closely at how customers behave, what they buy, and what they like. Then, they'll create a loyalty program with special rewards and benefits just for different groups of customers. This special treatment doesn't just keep customers coming back; it also makes them feel valued and part of the store's family, which makes them like the brand even more.

Considering everything mentioned above, the project objectives are:

1. Identify customer groups and create profiles for them. By doing this, it will be possible to assess common characteristics of client segments, such as product preferences and demographic information.
2. Design a loyalty program based on an ideal customer group with the intention of improving client retention and increasing revenue.
3. Achieve satisfactory financial results through customer segmentation and the loyalty program, which will be estimated in final steps.

By doing this, the business problem will be resolved.

# 5. Solution pipeline
The following pipeline was used, based on CRISP-DM framework:

1. Define the business problem.
2. Initial data understanding.
3. Exploratory data analysis and feature engineering (based on RFM model).
4. Data cleaning and preprocessing.
5. Group customers into clusters, modelling.
6. Analyze the groups created, profiling them (personas).
7. Develop the loyalty program.
8. Estimate financial results.

# 6. RFM model

<img src="reports/rfm.png">

I used the RFM model for clustering analysis. 

The RFM model is a marketing and customer segmentation technique used to analyze and categorize customers based on their recent purchasing behavior. RFM stands for:

- Recency: This measures how recently a customer has made a purchase. Customers who have made a purchase more recently are typically considered more valuable.
- Frequency: This measures how often a customer makes purchases. Customers who make frequent purchases are often more engaged and loyal.
- Monetary Value: This assesses the amount of money a customer has spent on purchases. Customers who have spent more are usually considered higher-value customers.

By analyzing these three factors, businesses can categorize their customers into different segments, such as "high-value and highly engaged" or "low-value and inactive." This segmentation allows companies to tailor their marketing strategies and offers to each group more effectively, ultimately improving customer retention and maximizing revenue.





| cluster          | monetary | frequency | recency | income | avg_purchase_value | numdealspurchases | numwebpurchases | numcatalogpurchases | numstorepurchases | numwebvisitsmonth | total_accepted_cmp | children | age  | relationship_duration | count | percentage |
|------------------|----------|-----------|---------|--------|--------------------|-------------------|-----------------|----------------------|-------------------|-------------------|--------------------|----------|------|-----------------------|-------|------------|
| Prosperous       | 1457.41  | 2.05      | 50.05   | 77641.28 | 74.02              | 1.06              | 4.89            | 6.07                 | 8.32              | 2.57              | 0.80               | 0.12     | 54.16 | 9.95                  | 476   | 21.36      |
| Web-Shrewd       | 996.90   | 2.30      | 49.34   | 64396.09 | 43.94              | 3.02              | 6.76            | 4.40                 | 9.06              | 5.03              | 0.34               | 0.94     | 57.92 | 10.13                 | 401   | 18.00      |
| Discount-Seeking | 454.81   | 1.81      | 48.05   | 49562.64 | 24.41              | 4.57              | 5.95            | 1.91                 | 6.01              | 6.71              | 0.20               | 1.39     | 56.78 | 10.23                 | 357   | 16.02      |
| Web Enthusiasts | 110.00   | 0.87      | 49.72   | 41588.22 | 11.59              | 1.81              | 2.05            | 0.69                 | 3.62              | 5.22              | 0.07               | 1.41     | 57.74 | 9.38                  | 465   | 20.87      |
| Young Budget     | 77.63    | 0.70      | 48.24   | 28574.70 | 10.37              | 1.86              | 1.94            | 0.42                 | 2.90              | 7.24              | 0.08               | 1.02     | 46.20 | 10.22                 | 529   | 23.74      |
