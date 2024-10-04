# 0.8742 / 5 horas 24 minutes / 0.84160
import gc
import ast
import time
import sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from tqdm import tqdm
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# start timer 
start_time = time.time()

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# List all the file names
file_names = ["ctr_15.csv", "ctr_16.csv", "ctr_17.csv", "ctr_18.csv", "ctr_19.csv", "ctr_20.csv", "ctr_21.csv"]

# Read and concatenate all the CSV files with progress tracking
train_data = pd.concat([pd.read_csv(file) for file in tqdm(file_names, desc="Loading CSV Files")])

#
random_state = 2345

# Sample data
train_data = train_data.sample(frac=1, random_state=random_state)

# Load the test data
test_data = pd.read_csv("ctr_test.csv")

# Run garbage collection after loading data
gc.collect()

##########################################################################################################################
##  FEATURE ENGINEERING
##########################################################################################################################

# Convert auction_time (Unix time) to datetime
train_data['auction_time'] = pd.to_datetime(train_data['auction_time'], unit='s')
test_data['auction_time'] = pd.to_datetime(test_data['auction_time'], unit='s')

# Create new features from auction_time
for df in tqdm([train_data, test_data], desc="Creating Auction Time Features"):
    df['auction_hour'] = df['auction_time'].dt.hour             # Hour of the auction
    df['auction_minute'] = df['auction_time'].dt.minute         # Minute of the auction
    df['auction_day_of_week'] = df['auction_time'].dt.dayofweek # Day of the week (Monday=0, Sunday=6)
    df['auction_day'] = df['auction_time'].dt.day               # Day of the month
    df['auction_month'] = df['auction_time'].dt.month           # Month of the year
    df['auction_year'] = df['auction_time'].dt.year             # Year of the auction
    df['is_weekend'] = df['auction_day_of_week'].apply(lambda x: 1 if x >= 5 else 0) #binary features for weekends
    
    df['auction_hour_bin'] = pd.cut(df['auction_hour'], bins=[0, 6, 12, 18, 24], labels=[1, 2, 3, 4]) #'night', 'morning', 'afternoon', 'evening'
    # Convert to categorical type
    df['auction_hour_bin'] = df['auction_hour_bin'].astype('category')

##################### good_hours

# Step 1: Identify the most popular hours (good hours)
auction_hour_counts = train_data['auction_hour'].value_counts().sort_values(ascending=False)
# Step 2: Define good hours (e.g., top 25% most frequent hours)
top_25_percent = int(0.25 * len(auction_hour_counts))
good_hours = auction_hour_counts.index[:top_25_percent].tolist()
# Step 3: Create the 'good_hours' feature
for df in tqdm([train_data, test_data], desc="Creating Good Hours Feature"):
    df['good_hours'] = df['auction_hour'].apply(lambda x: 1 if x in good_hours else 0)

   
##################### BINNING AND INTERACTIONS

# Binning auction_bidfloor into categories (to capture non linear effects)
train_data['bidfloor_binned'] = pd.cut(train_data['auction_bidfloor'], bins=[0, 1, 5, 10, 30], labels=[1, 2, 3, 4]) #'low', 'medium', 'high', 'very_high'
test_data['bidfloor_binned'] = pd.cut(test_data['auction_bidfloor'], bins=[0, 1, 5, 10, 30], labels=[1, 2, 3, 4]) #'low', 'medium', 'high', 'very_high'

##################### time since last auction interaction
for df in tqdm([train_data, test_data], desc="Calculating Time Since Last Auction"):
#     df['auction_time'] = pd.to_datetime(df['auction_time'])
    df['previous_auction_time'] = df.groupby('device_id')['auction_time'].shift(1)
    
    df['auction_time_unix'] = df['auction_time'].astype('int64') // 10**9
    df['previous_auction_time'] = df['previous_auction_time'].astype('int64') // 10**9
    
    df['time_since_last_auction_seconds'] = df['auction_time_unix'] - df['previous_auction_time']


for df in tqdm([train_data, test_data], desc="Calculating Rolling Bidfloor"):
    df['rolling_bidfloor'] = df.groupby('device_id')['auction_bidfloor'].transform(lambda x: x.rolling(window=7).mean())

##################### Reducing size 
# Group rare categories
threshold = 100  # You can adjust this threshold
value_counts = train_data['device_id_type'].value_counts()
rare_categories = value_counts[value_counts < threshold].index
train_data['device_id_type'] = train_data['device_id_type'].replace(rare_categories, 'Other')

# Now, dropping the original 'auction_time' since it's not needed anymore for the model
train_data = train_data.drop(['auction_time'], axis = 1)
test_data = test_data.drop(['auction_time'], axis = 1)

# Run garbage collection
gc.collect()

print("Train", train_data.columns)
print('Test', test_data.columns)
print("Data types for Train columns:", train_data.dtypes)
print("Data types for Test columns:", test_data.dtypes)

######################################################
# Columns Drop
######################################################
# Drop columns with more than 80% NAN
threshold = 0.8  # 80% threshold
missing_ratio = train_data.isnull().mean()
cols_to_drop_missing = missing_ratio[missing_ratio > threshold].index
print(f"Dropped columns missings: {cols_to_drop_missing}")
train_data.drop(cols_to_drop_missing, axis=1, inplace=True)
test_data.drop(cols_to_drop_missing, axis=1, inplace=True)

# # Dropping columns with high cardinality 
# # Function to determine if a column is unhashable
# def is_unhashable(column):
#     try:
#         # Try converting to a set to check for uniqueness
#         _ = set(column)
#         return False
#     except TypeError:
#         return True
# # Identify high cardinality columns, excluding unhashable types
# high_cardinality_cols = []
# for col in train_data.columns:
#     if train_data[col].dtype == 'object':
#         if is_unhashable(train_data[col]):
#             print(f"Skipping column {col} due to unhashable type.")
#             continue
#         if train_data[col].nunique() > 500:  
#             high_cardinality_cols.append(col)
# # Drop high cardinality columns from both train and test data
# train_data.drop(high_cardinality_cols, axis=1, inplace=True)
# test_data.drop(high_cardinality_cols, axis=1, inplace=True)
# print(f"Dropped columns high cardinality: {high_cardinality_cols}")

######################################################
## Model
######################################################

# Train a tree on the train data (sampling training data)
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])


## Calculate Scale Pos Weight
positive_class_count = y_train.value_counts().get(1, 0)
negative_class_count = y_train.value_counts().get(0, 0)
scale_pos_weight = negative_class_count / positive_class_count if positive_class_count > 0 else 1

# Train/test split for validation
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.15, random_state=random_state)


# Columns that contain lists (multi-categories)
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
numerical_cols = [col for col in X_train.columns if X_train[col].dtype != 'object']

# Convert object columns to categorical
for col in categorical_cols:
    X_train[col] = X_train[col].astype('category')

# Preprocessor for list columns (use MultiLabelBinarizer)
list_transformer = MultiLabelBinarizer()

# Define a column transformer to handle categorical and numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numerical_cols),  # Impute and scale numerical data
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),  # One-hot encode categorical data
    ]
)



hyperparameters = {'colsample_bytree': 0.6509900925068455, 'gamma': 1.1805850841291976, 'learning_rate': 0.009732823220209536,
          'max_depth': 8, 'min_child_weight': 63.01501268012936, 'n_estimators': 2500, 'reg_lambda': 1.2219990638829437,
          'subsample': 0.9692112596805615, 'scale_pos_weight': scale_pos_weight}


model = xgb.XGBClassifier(objective='binary:logistic', seed=random_state, eval_metric='auc', **hyperparameters, tree_method='hist')


gc.collect()




######################################################
## file
######################################################

# K_FOLD = 0  # If 0 use HOLDOUT 0.15, else N FOLD CV
# preds = pd.DataFrame()  # Contains all predictions
# scores = []


# if K_FOLD == 0:
#     # HOLDOUT
#     model.fit(X_train_split, y_train_split)
#     temp_preds = model.predict_proba(X_valid_split)[:, 1]
#     preds = pd.concat([preds, pd.DataFrame(temp_preds)], axis=1)
#     roc_auc = sklearn.metrics.roc_auc_score(y_valid_split, temp_preds)
#     print(f"ROC AUC Score for HOLDOUT SET: {roc_auc}")
#     scores.append(roc_auc)

#     # YPREDS AVERAGE AND CHECK ON VALIDATION
#     final_ypred = np.mean(preds, axis=1)

#     # Calculate the final ROC AUC score using the average predictions
#     final_roc_auc = sklearn.metrics.roc_auc_score(y_valid_split, final_ypred)
#     print(f"Final ROC AUC Score (Average of Predictions) for HOLDOUT SET: {final_roc_auc}")

# else:
#     # K-FOLD Cross Validation
#     kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=random_state)

#     # Perform cross-validation and get the cross-validated ROC AUC scores
#     temp_preds = cross_val_predict(model, X_train_split, y_train_split, cv=kf, method='predict_proba')[:, 1]
#     preds = pd.concat([preds, pd.DataFrame(temp_preds)], axis=1)
#     roc_auc = sklearn.metrics.roc_auc_score(y_train_split, temp_preds)
#     print(f"ROC AUC Score for K-FOLD CV: {roc_auc}")
#     scores.append(roc_auc)

#     # YPREDS AVERAGE AND CHECK ON TRAINING DATA (since all data was used for training in different steps)
#     final_ypred = np.mean(preds, axis=1)

#     # Calculate the final ROC AUC score using the average predictions
#     final_roc_auc = sklearn.metrics.roc_auc_score(y_train_split, final_ypred)
#     print(f"Final ROC AUC Score (Average of Predictions) for KFOLD CV: {final_roc_auc}")

print(f"Training model")
    
# Define the pipeline for this model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
        ('classifier', model)
])
    
# Fit the model
pipeline.fit(X_train_split, y_train_split)
    
# Predict on the validation set
y_valid_preds = pipeline.predict_proba(X_valid_split)[:, 1]

# Evaluate model performance (e.g., AUC score)
auc_score = roc_auc_score(y_valid_split, y_valid_preds)
print(f"Model AUC: {auc_score:.4f}")

gc.collect()

######################################################
## file
######################################################

# final_preds = test_data.drop(columns=["id"])
# preds = pd.DataFrame()

# print("Training Model on Full Data")
# model.fit(X_train, y_train)
# temp_preds = model.predict_proba(test_data.drop(columns=["id"]))[:, 1]
# preds = pd.concat([preds, pd.DataFrame(temp_preds)], axis=1)

# # Average predictions and reset index
# final_ypred = np.mean(preds, axis=1).reset_index(drop=True)
# test_data = test_data.reset_index(drop=True)

# # Make the submission file
# submission_df = pd.DataFrame({"id": test_data["id"], "Label": final_ypred})
# submission_df["id"] = submission_df["id"].astype(int)
# submission_df.to_csv("lgbm_model.csv", sep=",", index=False)

print(f"Retraining model on the full dataset")
# Fit the model on the full dataset
pipeline.fit(X_train, y_train)
# Step 2: Generate Predictions for the Test set
print(f"Preparing final submission file")
# Drop the 'id' column as it's not a feature
X_test = test_data.drop(columns=["id"])
# Generate predictions for the test set
y_preds_final = pipeline.predict_proba(X_test)[:, 1]

# Step 2.5: Create the Submission DataFrame
submission_df = pd.DataFrame({"id": test_data["id"],"Label": y_preds_final})
submission_df["id"] = submission_df["id"].astype(int)
# Step 2.6: Save the Submission File
submission_df.to_csv("basic_model_xgboost.csv", sep=",", index=False)
print("Submission file 'basic_model_xboost.csv' created successfully.")


######################################################
## timer end
######################################################

# End timer
# Calculate the elapsed time in hours, minutes, and seconds
elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

# Print the elapsed time
print("Process finished --- %d hours %d minutes %d seconds --- " % (hours, minutes, seconds))

