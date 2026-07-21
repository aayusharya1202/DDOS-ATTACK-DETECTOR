import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import numpy as np
full_df = pd.read_parquet(r'C:\DATASET\DATA\cicids2017_combined.parquet')
print("File loaded successfully.")
print(f"Initial DataFrame shape: {full_df.shape}")
print("\n--- Starting Data Cleaning and Preprocessing ---")

initial_rows = full_df.shape[0]
full_df.drop_duplicates(inplace=True)
dropped_rows = initial_rows - full_df.shape[0]
print(f"\nStep 1: Dropping duplicate rows")
print(f"Initial rows: {initial_rows}")
print(f"Dropped {dropped_rows} duplicate rows.")
print(f"New DataFrame shape: {full_df.shape}")

missing_values_count = full_df.isnull().sum()
total_rows = len(full_df)
missing_values_percent = (missing_values_count / total_rows) * 100
columns_to_drop = missing_values_percent[missing_values_percent > 10].index.tolist()

numerical_cols = full_df.select_dtypes(include=np.number).columns.tolist()
correlation_matrix = full_df[numerical_cols].corr().abs()
print("\nStep 2: Creating correlation matrix for numerical features")
print("Correlation Matrix of Numerical Features (first 5x5):")
print(correlation_matrix.head(5).iloc[:, :5])

print("\nStep 3: Dropping highly correlated columns (> 0.95)")
correlated_cols_to_drop = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if correlation_matrix.iloc[i, j] > 0.95:
            colname = correlation_matrix.columns[i]
            if colname not in correlated_cols_to_drop:
                correlated_cols_to_drop.add(colname)

full_df.drop(columns=correlated_cols_to_drop, inplace=True)
print(f"Dropped {len(correlated_cols_to_drop)} highly correlated columns.")
print(f"New DataFrame shape after dropping correlated columns: {full_df.shape}")

final_num_features = full_df.drop(['Label'], axis=1).select_dtypes(include='number').shape[1]
print(f"\nTotal number of features remaining for the model: {final_num_features}")

print("\nStarting Model Training and Evaluation ")
print("\nStep 4: Encoding the target variable 'Label'")
print("Unique labels before encoding:")
print(full_df['Label'].value_counts())
label_encoder = LabelEncoder()
full_df['Label_encoded'] = label_encoder.fit_transform(full_df['Label'])
y = full_df['Label_encoded']

X = full_df.drop(['Label', 'Label_encoded'], axis=1)
X = X.select_dtypes(include='number') 
print("\nFeatures (X) and Target (y) prepared.")
print(f"Features DataFrame shape (X): {X.shape}")
print(f"Target Series shape (y): {y.shape}")
print("The first few rows of features (X):")
print(X.head())
print("\nThe first few rows of the encoded target (y):")
print(y.head())


print("\nStep 5: Splitting data into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set size: {X_test.shape[0]} samples")

print("\nStep 6: Applying SMOTE for oversampling on the training data")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("SMOTE oversampling complete.")
print(f"Original training set size: {X_train.shape[0]} samples")
print(f"Resampled training set size: {X_train_res.shape[0]} samples")
print("\nDistribution of labels in resampled training set:")
print(pd.Series(y_train_res).value_counts())

# Train the Random Forest model
print("\nStep 7:Training the Random Forest model on resampled data")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_res, y_train_res)
print("Model training complete.")

print("\nStep 8 : Making predictions on the test set")
y_pred = rf_model.predict(X_test)


print("\nStep 9 : Evaluating the model")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
