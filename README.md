# Machine Learning Data Pipeline

#### Video Demo:
https://youtu.be/S0M90A9QvM4

## Description

This project is a reusable machine learning pipeline built in Python that analyzes the **Heart Disease** csv dataset from the UC Irvine Machine Learning Repository. It automates common steps in a machine learning workflow, including loading data, cleaning missing values, separating features and targets, splitting data into training and testing sets, scaling features, training a classification model, predicting heart disease in new patients, and evaluating model performance.

The goal of this project is to create a simple framework that can predict heart disease in new patient data and making a machine learning workflow that is organized and easy to reuse across different chronic disease datasets.

## Features

- Loads datasets from CSV files using pandas
- Removes missing values from datasets
- Separates feature variables from the target variable
- Splits data into training and testing sets
- Standardizes numerical features using StandardScaler
- Trains a logistic regression classification model
- Calculates model accuracy
- Generates classification reports and confusion matrices
- Saves and loads trained machine learning systems

## Project Structure
project/
│
├── project.py # Main program containing classes
├── requirements.txt # Required Python packages
└── README.md # Project documentation


## Design

The project is organized into three main classes:

### DataPipeline

Handles all data preparation steps:

- Loading data
- Cleaning missing values
- Creating feature and target datasets
- Splitting data
- Scaling features

### ModelTrainer

Handles machine learning model training and evaluation:

- Training the model
- Measuring accuracy
- Creating evaluation reports
- Generating confusion matrices

### MLSystem

Combines a trained model and scaler into one reusable object:

- Making predictions on new data
- Saving the trained system
- Loading a saved system

## Installation

Install the required packages: pip install pandas numpy scikit-learn joblib


## Usage

Example workflow:

```python
pipeline = DataPipeline(
    "data.csv",
    "target"
)

pipeline.load_data()
pipeline.clean_data()
pipeline.split_features_target()
pipeline.split_data()
pipeline.scale_features()

trainer = ModelTrainer()

trainer.train_model(
    pipeline.X_train,
    pipeline.y_train
)

accuracy = trainer.evaluate(
    pipeline.X_test,
    pipeline.y_test
)

print(accuracy)

Technologies Used:

Python
pandas
NumPy
scikit-learn
joblib

Possible Improvements Include:

Supporting multiple machine learning algorithms
Adding automatic feature encoding for categorical variables
Adding hyperparameter tuning
Creating a graphical user interface
