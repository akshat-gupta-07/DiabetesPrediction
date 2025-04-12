# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Loading the dataset
df = pd.read_csv(r'kaggle_diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from specified columns with NaN
df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
].replace(0, np.nan)

# Replacing NaN values using mean or median depending on distribution
df_copy['Glucose'] = df_copy['Glucose'].fillna(df_copy['Glucose'].mean())
df_copy['BloodPressure'] = df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean())
df_copy['SkinThickness'] = df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median())
df_copy['Insulin'] = df_copy['Insulin'].fillna(df_copy['Insulin'].median())
df_copy['BMI'] = df_copy['BMI'].fillna(df_copy['BMI'].median())

# Updating df with cleaned data
df = df_copy

# Dropping 'Pregnancies' column before training
df = df.drop(columns='Pregnancies')

# Splitting dataset into features and target
X = df.drop(columns='Outcome')
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating and training the Random Forest Classifier model
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Define the correct relative path to the 'model' directory in the parent folder
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
os.makedirs(model_dir, exist_ok=True)

# Saving the trained model using pickle
filename = os.path.join(model_dir, 'diabetes-model.pkl')
pickle.dump(classifier, open(filename, 'wb'))

print("SUCCESS")
