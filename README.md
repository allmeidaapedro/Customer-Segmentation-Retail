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








| cluster          | monetary | frequency | recency | income | avg_purchase_value | numdealspurchases | numwebpurchases | numcatalogpurchases | numstorepurchases | numwebvisitsmonth | total_accepted_cmp | children | age  | relationship_duration | count | percentage |
|------------------|----------|-----------|---------|--------|--------------------|-------------------|-----------------|----------------------|-------------------|-------------------|--------------------|----------|------|-----------------------|-------|------------|
| Prosperous       | 1457.41  | 2.05      | 50.05   | 77641.28 | 74.02              | 1.06              | 4.89            | 6.07                 | 8.32              | 2.57              | 0.80               | 0.12     | 54.16 | 9.95                  | 476   | 21.36      |
| Web-Shrewd       | 996.90   | 2.30      | 49.34   | 64396.09 | 43.94              | 3.02              | 6.76            | 4.40                 | 9.06              | 5.03              | 0.34               | 0.94     | 57.92 | 10.13                 | 401   | 18.00      |
| Discount-Seeking | 454.81   | 1.81      | 48.05   | 49562.64 | 24.41              | 4.57              | 5.95            | 1.91                 | 6.01              | 6.71              | 0.20               | 1.39     | 56.78 | 10.23                 | 357   | 16.02      |
| Web Enthusiasts | 110.00   | 0.87      | 49.72   | 41588.22 | 11.59              | 1.81              | 2.05            | 0.69                 | 3.62              | 5.22              | 0.07               | 1.41     | 57.74 | 9.38                  | 465   | 20.87      |
| Young Budget     | 77.63    | 0.70      | 48.24   | 28574.70 | 10.37              | 1.86              | 1.94            | 0.42                 | 2.90              | 7.24              | 0.08               | 1.02     | 46.20 | 10.22                 | 529   | 23.74      |
