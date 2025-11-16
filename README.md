# 🚗 New York Car Price Prediction

### *A Complete Machine Learning Pipeline for Predicting Used Car Prices in New York*

This repository contains a full end-to-end machine learning project for
predicting used car prices using multiple real-world automotive
datasets. It includes raw data, cleaning scripts, preprocessing steps,
visualizations, and model training inside a Jupyter Notebook.

The project aims to accurately estimate selling prices based on key
vehicle attributes, helping buyers, sellers, and dealerships understand
price trends in the New York market.

------------------------------------------------------------------------

# 🌟 Project Banner

    ███████╗██╗   ██╗██████╗ ██████╗ ██████╗ ██████╗ █████╗ ██████╗ 
    ██╔════╝██║   ██║██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗
    █████╗  ██║   ██║██████╔╝██████╔╝██████╔╝██████╔╝███████║██████╔╝
    ██╔══╝  ██║   ██║██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██║██╔══██╗
    ██║     ╚██████╔╝██║  ██║██║  ██║██║  ██║██║  ██║██║  ██║██║  ██║
    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝

------------------------------------------------------------------------

# 📂 Repository Structure

    📦 New-York-Car-Price-Prediction
    │
    ├── data/
    │   ├── Car_Rates.csv
    │   ├── New_York_cars (1).csv
    │   ├── vehicles (1).csv
    │   └── newyork_car_price_preprocessed.xls
    │
    ├── notebooks/
    │   └── NEW YORK CAR PRICE PREDICTION.ipynb
    │
    ├── models/
    │   └── (optional saved models)
    │
    ├── README.md
    ├── requirements.txt
    └── LICENSE

------------------------------------------------------------------------

# 📝 Project Description

The used-car market in New York has thousands of listings across various
sources. Predicting car prices is challenging due to:

-   Varying conditions\
-   Model-year depreciation\
-   Brand-based price tiers\
-   Mileage differences\
-   Fuel type & transmission\
-   Seasonal and regional variations

This project builds a robust ML model capable of learning these patterns
using a large, combined dataset from multiple CSV files.

------------------------------------------------------------------------

# 📊 Datasets Overview

### **1. New_York_cars (1).csv**

Primary dataset containing: - Price\
- Year\
- Manufacturer\
- Model\
- Condition\
- Cylinders\
- Fuel\
- Odometer\
- Transmission\
- Drive type\
- Body type\
- VIN\
- Location info

### **2. vehicles (1).csv**

Supplementary dataset used to enrich model accuracy.

### **3. Car_Rates.csv**

Contains depreciation and value-adjustment factors.

### **4. newyork_car_price_preprocessed.xls**

Final cleaned and model-ready dataset.

------------------------------------------------------------------------

# 🧹 Data Cleaning & Preprocessing

-   Removal of duplicates and irrelevant rows\
-   Handling missing values using median/mode\
-   Fixing inconsistencies across manufacturers and models\
-   Outlier detection using IQR and Z-score\
-   Feature engineering including:
    -   Vehicle Age\
    -   Engine Grouping\
    -   Brand Demand Categories\
    -   Price-per-mile ratios\
-   Encoding categorical features\
-   Scaling numerical columns\
-   Train-test split (80/20)

------------------------------------------------------------------------

# 🤖 Machine Learning Models

Models evaluated:

-   Linear Regression\
-   Random Forest Regressor\
-   XGBoost Regressor\
-   ExtraTrees Regressor\
-   Ridge & Lasso Regression

Evaluation metrics include MAE, MSE, RMSE, and R².

------------------------------------------------------------------------

# 📈 Visualizations

The notebook includes:

-   Correlation heatmap\
-   Price distribution analysis\
-   Scatter plots\
-   Feature importance charts\
-   Actual vs Predicted comparison

------------------------------------------------------------------------

# 🛠️ Installation & Usage

### Install dependencies:

    pip install -r requirements.txt

### Run notebook:

    jupyter notebook "NEW YORK CAR PRICE PREDICTION.ipynb"

------------------------------------------------------------------------

# 📦 requirements.txt (included in README)

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    xgboost
    jupyter
    openpyxl
    joblib

------------------------------------------------------------------------

# 🏁 Conclusion

This project successfully demonstrates the application of machine
learning techniques to predict used car prices in the New York market
with a high degree of accuracy. By integrating multiple datasets,
performing rigorous data cleaning, and applying effective feature
engineering, the final model is able to capture key pricing patterns
across manufacturers, vehicle conditions, model years, mileage levels,
and technical specifications.

Through comprehensive exploratory analysis and evaluation of several
regression algorithms, the project highlights the factors that most
strongly influence car prices and provides a reliable predictive
framework for estimating fair market values. The trained
models---especially ensemble methods such as Random Forest and
XGBoost---show strong performance and practical usability for real-world
pricing scenarios.

Overall, this work provides a solid foundation for automated vehicle
valuation systems and can be extended further through model deployment,
hyperparameter optimization, and integration with live data sources. The
project not only showcases a complete end-to-end machine learning
pipeline but also delivers meaningful insights that can support buyers,
sellers, and automotive platforms in making informed decisions.

------------------------------------------------------------------------

# 🏷️ GitHub Tags / Keywords

    machine-learning
    car-price-prediction
    data-science
    python
    xgboost
    regression-models
    vehicle-analytics
    new-york-dataset
    data-cleaning
    ml-project

------------------------------------------------------------------------

# 📜 License

MIT License
