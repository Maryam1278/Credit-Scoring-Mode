import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load the dataset (correct the file path)
df = pd.read_csv(r"C:\Users\mariu\OneDrive\Desktop\work\Intership\loan_approval_dataset.csv")

# Display the first few rows of the dataset
df.head()
# Display the first few rows of the dataset
print("Dataset Loaded Successfully!")
print(df.head())
# Check the columns in the DataFrame
print("Columns in the DataFrame:", df.columns.tolist())

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Fill missing values (example: filling with mode for categorical and mean for numerical)
if 'self_employed' in df.columns:
    df['self_employed'].fillna(df['self_employed'].mode()[0], inplace=True)

if 'loan_amount' in df.columns:
    df['loan_amount'].fillna(df['loan_amount'].mean(), inplace=True)

# Encode categorical variables using Label Encoding
label_encoders = {}
categorical_cols = ['education', 'self_employed']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Define features and target variable
# Corrected target variable to 'loan_status' after stripping spaces
if 'loan_status' in df.columns:
    X = df.drop('loan_status', axis=1)  # Features
    y = df['loan_status']  # Target variable
else:
    print("Error: 'loan_status' column not found in the dataset.")
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Assess the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Display classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
