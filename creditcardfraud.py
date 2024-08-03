

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('creditcard.csv')

# Inspect the dataset
print(df.head())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Check the class distribution
print(df['Class'].value_counts())

# Data preprocessing
# Standardizing the 'Amount' feature and dropping irrelevant columns
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(columns=['Time'], inplace=True)

# Splitting the data into features and labels
X = df.drop(columns=['Class'])
y = df['Class']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handling the imbalanced dataset using SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

### Step 2: Exploratory Data Analysis (EDA)

# Visualize the class distribution
sns.countplot(x='Class', data=df)
plt.title('Class Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

### Step 3: Model Selection and Training

# Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_res, y_train_res)

### Step 4: Model Evaluation

# Predictions
y_pred = rf_classifier.predict(X_test)

# Model evaluation
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nClassification Report:')
print(classification_report(y_test, y_pred))

print('\nAccuracy Score:')
print(accuracy_score(y_test, y_pred))


