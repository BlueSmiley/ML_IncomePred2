# half these imports arent used anymore and can be safely removed
import pandas as pd
import numpy as np
from sklearn import  linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.feature_selection import RFECV
from category_encoders import LeaveOneOutEncoder, TargetEncoder,OrdinalEncoder,CatBoostEncoder
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn import neural_network
import xgboost as xgb
import lightgbm as lgb
import re
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC

def main():
    # Load input and split into train, test and validation
    input_file = "tcd-ml-1920-group-income-train.csv"
    data = pd.read_csv(input_file, header = 0)
    print("Starting")
    train, validation = train_test_split(data, test_size=0.20)
    train, test = train_test_split(train, test_size=0.2)
    
    #Preprocess on train using target encoding
    (train,encoder) = clean_train_data_target_encoded(train)
    
    # load final data and preprocess it using trained encoder
    actual_file = "tcd-ml-1920-group-income-test.csv"
    finalData = pd.read_csv(actual_file, header = 0)
    finalData = finalData.iloc[:,:-1]
    finalData = clean_data(finalData, encoder)
    
    # preprocessing for train and validation
    train_y = train.iloc[:,-1]
    train_X = train.iloc[:,:-1]
    test_y = test.iloc[:,-1]
    test_X = test.iloc[:,:-1]
    test_X = clean_data(test_X, encoder)
    val_y = validation.iloc[:,-1]
    val_X = validation.iloc[:,:-1]
    val_X = clean_data(val_X, encoder)
    
    
    print("training data cleaned")
            
    # Convert to usable validation format for lightboost
    # All settings actually default here excpet for learning rate, metric
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train)
    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'mae'},
        'learning_rate': 0.01,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'num_leaves': 31,
        'verbose': 0
    }
    
    # Num boost rounds put to 100k but finishes around 40-80k anyway
    # Ealry stopping round of 200 to avoid local minima
    print('Starting training...')
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100000,
                valid_sets=lgb_eval,
                early_stopping_rounds=200)
    print('Starting predicting...')
    
    # Calculate and print out stats for local validation set
    pred = gbm.predict(val_X, num_iteration=gbm.best_iteration)
    print("Root Mean squared error: %.2f"
        %  sqrt(mean_squared_error(val_y, pred)))
    
    print("Mean absolute error: %.2f"
        %  mean_absolute_error(val_y, pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(val_y, pred))
    
     # Calculate final predictions for actual dataset and write to file
    results = gbm.predict(finalData, num_iteration=gbm.best_iteration)
    output_file = "tcd-ml-1920-group-income-submission.csv"
    output =  pd.read_csv(output_file, header = 0, index_col=False)
    output["Total Yearly Income [EUR]"] = results
    output.to_csv(output_file, index=False)
    print("We done :)")
    

# Code to preprocess using a trained encoder
def clean_data(data, encoder):
    """Clean the final data using the given one hot encoder"""
    data = data.reset_index(drop=True)
    data = process_features(data) 
    data2 = encoder.transform(data)   
    #data2 = data2.fillna(method="ffill")
    return data2

# Code to preprocess and train encoder on data
def clean_train_data_target_encoded(data):
    #uses target encodier instead
    data = data.reset_index(drop=True)
    train_y = data.iloc[:,-1]
    train_y = train_y.reset_index(drop=True)
    train_X = data.iloc[:,:-1]
    
    train_X = process_features(train_X)
    
    
    encoder = TargetEncoder(cols = ["Hair Color",
         "Wears Glasses","University Degree","Gender","Country","Profession", 
         "Housing Situation", "Satisfation with employer"], smoothing = 300)

    encoder.fit(train_X,train_y)
    data2 =  pd.concat([encoder.transform(train_X,train_y).reset_index(drop=True),train_y.reset_index(drop=True)],axis=1)
    #data2 = data2.fillna(method="ffill")
    
    return (data2,encoder)

# Preprocessing steps for data. Mostly just fill in na with mean or new category
# Depending on type of column
def process_features(data):
    data = data.drop('Instance', 1)
    data["Hair Color"] = data["Hair Color"].fillna("Unknown")
    data["Hair Color"] = data["Hair Color"].astype('str')
    data["University Degree"] = data["University Degree"].fillna("Unknown")
    data["University Degree"] = data["University Degree"].astype('str')
    data["Gender"] = data["Gender"].fillna("Unknown")
    data["Gender"] = data["Gender"].astype('str')
    data["Country"] = data["Country"].fillna("Unknown")
    data["Country"] = data["Country"].astype('str')
    data["Satisfation with employer"] = data["Satisfation with employer"].fillna("Unknown")
    data["Satisfation with employer"] = data["Satisfation with employer"].astype('str')
    data["Profession"] = data["Profession"].fillna("Unknown")
    data["Profession"] = data["Profession"].astype('str')
    data["Housing Situation"] = data["Housing Situation"].fillna("Unknown")
    data["Housing Situation"] = data["Housing Situation"].astype('str')
    
    data['Yearly Income in addition to Salary (e.g. Rental Income)'] = (
            data['Yearly Income in addition to Salary (e.g. Rental Income)'].
            apply(lambda x: re.search(r'\d+', x).group()).astype(int))
    data["Work Experience in Current Job [years]"] = (
            pd.to_numeric(
            data["Work Experience in Current Job [years]"], errors='coerce'))
    data["Crime Level in the City of Employement"] = (
            pd.to_numeric(
            data["Crime Level in the City of Employement"], errors='coerce'))
    
    # For numeric rows
    print("Filling means")
    data["Crime Level in the City of Employement"] = (
            data["Crime Level in the City of Employement"].fillna(
                    data["Crime Level in the City of Employement"].mean()))
    data["Year of Record"] = (
            data["Year of Record"].fillna(
                    data["Year of Record"].mean()))
    data["Work Experience in Current Job [years]"] = (
            data["Work Experience in Current Job [years]"].fillna(
                    data["Work Experience in Current Job [years]"].mean()))
    data["Age"] = (
            data["Age"].fillna(
                    data["Age"].mean()))
    data["Size of City"] = (
            data["Size of City"].fillna(
                    data["Size of City"].mean()))
    data["Body Height [cm]"] = (
            data["Body Height [cm]"].fillna(
                    data["Body Height [cm]"].mean()))
    data["Yearly Income in addition to Salary (e.g. Rental Income)"] = (
            data["Yearly Income in addition to Salary (e.g. Rental Income)"].fillna(
            data["Yearly Income in addition to Salary (e.g. Rental Income)"].mean()))
    print("Mean filling done")
    return data
    
if __name__ == "__main__":
   main()