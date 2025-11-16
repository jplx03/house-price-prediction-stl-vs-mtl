# house-price-prediction-stl-vs-mtl
323 group project on house price prediction

dataset: kc_house_data.csv file taken from kaggle site: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/data 

Overview: 
The project evaluates house price prediction performance under three learning approaches:
1. Ridge Regression (STL)
2. Random Forest (STL)
3. L2,1-regularised MTL

The 'House Sales in King County, USA' dataset is used, with additional preprocessing and feature engineering. The goal is to compare STL vs MTL under consistent evaluation setup. 

## Environment setup
required libraries:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- torch

to load the dataset directly from github:
'''import pandas as pd
url = "https://raw.githubusercontent.com/jplx03/house-price-prediction-stl-vs-mtl/main/dataset/kc_house_data.csv"
df = pd.read_csv(url)
'''
