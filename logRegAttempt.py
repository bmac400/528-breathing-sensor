import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Directory path
directoryArr = ['Data/Abnormal', 'Data/Normal']

# Get all files in the directory with absolute paths


dfs = []
labels = []
# Process each file
for directory in directoryArr:
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    for f in files:
        # do something
        if "Sternum" in f:
            continue
        
        df = pd.read_csv(f)
        if "time" in df:
            df.drop(columns=['time'])
        if directory == "Data/Abnormal":
            labels.append("Abnormal")
        else:
            labels.append("Normal")

        # Transpose DataFrame
        df_transposed = df.transpose()

        # Reset index to get original column names as a new column
        df_transposed.reset_index(inplace=True)

        # Rename columns
        new_columns = {i: f'{col}{i+1}' for i, col in enumerate(df_transposed.iloc[:, 0])}
        df_transposed = df_transposed.rename(columns=new_columns)

        # Drop the first column (original column names)
        df_transposed.drop(columns='index', inplace=True)

        # Keep only the first row, as we transposed the DataFrame
        df_transposed = df_transposed.iloc[0, :]

        # Convert to DataFrame with a single row
        df_result = pd.DataFrame(df_transposed).transpose()
        dfs.append(df_result)
        # Split data into features (X) and labels (y)
X = pd.concat(dfs, axis=0)  # Assuming 'label' is the name of your target column
y = labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Train logistic regression model
classifier = LogisticRegression(max_iter=100000)
classifier.fit(X_train_scaled, y_train)

# Evaluate logistic regression model
y_pred_lr = classifier.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", accuracy_lr)

# Train random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Evaluate random forest classifier
y_pred_rf = rf_classifier.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)