# Car-Resale_Price Analysis and Prediction Using XGBoost

## Overview

This project aims to analyze and predict car prices using various machine learning techniques. The dataset contains information about cars, including their maker, model year, distance covered, city, fuel type, and more. The goal is to train a model that can accurately predict the price of a car based on these features.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Visualization](#visualization)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)
- [License](#license)

## Project Description

The project involves the following steps:

1. *Data Preprocessing:* Clean the dataset by handling missing values, converting categorical variables to numerical values, and formatting the data.
2. *Exploratory Data Analysis (EDA):* Perform various visualizations to understand the data distribution and relationships between different variables.
3. *Model Building:* Use the XGBoost Regressor to predict car prices. The model is trained on 80% of the dataset, with 20% reserved for testing.
4. *Hyperparameter Tuning:* Utilize GridSearchCV to find the best hyperparameters for the XGBoost model.
5. *Model Evaluation:* Assess the performance of the model using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).

## Dataset

The dataset Cars.csv contains the following columns:

- maker: The car manufacturer.
- model_name: The model of the car.
- model_year: The year the car model was released.
- distance_covered (km): The total distance the car has covered.
- city: The city where the car is being sold.
- fuel_type: The type of fuel the car uses (e.g., Petrol, Diesel).
- pre_owner: The number of previous owners.
- price (₹): The price of the car in Indian Rupees.

## Data Preprocessing

Data preprocessing steps include:

1. Handling missing values by checking for null values.
2. Converting the model_year column to datetime format.
3. Cleaning the price (₹) column by removing the currency symbol and converting the values to float.
4. Label encoding the categorical variables for model training.

## Exploratory Data Analysis

The following analyses and visualizations were performed:

- *Count of Cars by Model Year:* A bar chart showing the number of cars for each model year.
- *Count of Cars by Maker:* A bar chart displaying the number of cars from each manufacturer.
- *Percentage Share of Car Makers:* A pie chart representing the market share of different car makers.
- *Top 10 Car Models by Count:* A bar chart showing the most common car models in the dataset.
- *Distribution of Distance Covered:* A histogram displaying the distribution of the distance covered by cars.
- *Box Plot of Distance Covered:* A box plot showing the spread of distances covered by cars.
- *Count of Cars by Fuel Type:* A bar chart displaying the count of cars by fuel type.
- *Count of Cars by Number of Previous Owners:* A bar chart showing the number of cars by the number of previous owners.
- *Distribution of Car Prices:* A histogram showing the distribution of car prices.

## Model Building

The XGBoost Regressor was used to predict the car prices. The model was trained on 80% of the data, and the remaining 20% was used for testing.

## Hyperparameter Tuning

GridSearchCV was employed to fine-tune the hyperparameters of the XGBoost model. The parameters tuned include:

- n_estimators
- learning_rate
- max_depth
- subsample
- colsample_bytree

## Model Evaluation

The performance of the model was evaluated using the following metrics:

- *Mean Absolute Error (MAE):* Measures the average magnitude of the errors in a set of predictions.
- *Mean Squared Error (MSE):* Measures the average of the squares of the errors.
- *Root Mean Squared Error (RMSE):* The square root of the average of squared differences between prediction and actual observation.
- *R-squared (R²):* Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Visualization

The project also includes a visualization that compares the actual vs. predicted car prices using a line plot.

## Requirements

The following Python libraries are required to run the project:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
