import gc
import time
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint


# start timer 
start_time = time.time()

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# List all the file names
file_names = ["ctr_15.csv", "ctr_16.csv", "ctr_17.csv", "ctr_18.csv", "ctr_19.csv", "ctr_20.csv", "ctr_21.csv"]

# Read and concatenate all the CSV files
train_data = pd.concat([pd.read_csv(file) for file in tqdm(file_names, desc="Reading CSV files")])

random_state=2345
# Sample data
train_data = train_data.sample(frac=1, random_state=random_state)

# Load the test data
test_data = pd.read_csv("ctr_test.csv")

##########################################################################################################################
##  FEATURE ENGINEERING
##########################################################################################################################

# Convert auction_time (Unix time) to datetime
train_data['auction_time'] = pd.to_datetime(train_data['auction_time'], unit='s')
test_data['auction_time'] = pd.to_datetime(test_data['auction_time'], unit='s')

# Create new features from auction_time
for df in tqdm([train_data, test_data], desc="Creating auction time features"):
    df['auction_hour'] = df['auction_time'].dt.hour             # Hour of the auction
    df['auction_minute'] = df['auction_time'].dt.minute         # Minute of the auction
    df['auction_day_of_week'] = df['auction_time'].dt.dayofweek # Day of the week (Monday=0, Sunday=6)
    df['auction_day'] = df['auction_time'].dt.day               # Day of the month
#     df['auction_month'] = df['auction_time'].dt.month           # Month of the year
#     df['auction_year'] = df['auction_time'].dt.year             # Year of the auction
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
for df in tqdm([train_data, test_data], desc="Creating good_hours feature"):
    df['good_hours'] = df['auction_hour'].apply(lambda x: 1 if x in good_hours else 0)

##################### action_categorical_*: 

# Combine different levels of business unit IDs
for df in tqdm([train_data, test_data], desc="Combining action categorical features"):
    df['action_cat_0_1'] = df['action_categorical_0'].astype(str) + '_' + df['action_categorical_1'].astype(str)
    df['action_cat_1_3'] = df['action_categorical_1'].astype(str) + '_' + df['action_categorical_3'].astype(str)
    df['action_cat_3_5'] = df['action_categorical_3'].astype(str) + '_' + df['action_categorical_5'].astype(str)
    df['action_cat_5_6'] = df['action_categorical_5'].astype(str) + '_' + df['action_categorical_6'].astype(str)
   
##################### BINNING AND INTERACTIONS

# Binning auction_bidfloor into categories (to capture non linear effects)
train_data['bidfloor_binned'] = pd.cut(train_data['auction_bidfloor'], bins=[0, 1, 5, 10, 30], labels=[1, 2, 3, 4]) #'low', 'medium', 'high', 'very_high'
test_data['bidfloor_binned'] = pd.cut(test_data['auction_bidfloor'], bins=[0, 1, 5, 10, 30], labels=[1, 2, 3, 4]) #'low', 'medium', 'high', 'very_high'

##################### time since last auction interaction
for df in tqdm([train_data, test_data], desc="Calculating time since last auction"):
#     df['auction_time'] = pd.to_datetime(df['auction_time'])
    df['previous_auction_time'] = df.groupby('device_id')['auction_time'].shift(1)
    
    df['auction_time_unix'] = df['auction_time'].astype('int64') // 10**9
    df['previous_auction_time'] = df['previous_auction_time'].astype('int64') // 10**9
    
    df['time_since_last_auction_seconds'] = df['auction_time_unix'] - df['previous_auction_time']


for df in tqdm([train_data, test_data], desc="Rolling bidfloor calculation"):
    df['rolling_bidfloor'] = df.groupby('device_id')['auction_bidfloor'].transform(lambda x: x.rolling(window=1).mean())

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

# Run garbage collection
gc.collect()

######################################################
## Model
######################################################
print("Data types for Train columns:", train_data.dtypes)
print("Data types for Test columns:", test_data.dtypes)

# Train a tree on the train data (sampling training data)
# train_data = train_data.sample(frac=8/10)
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])

# Train/test split for validation
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.15, random_state=random_state, stratify=y_train)

# Separate the categorical and numerical columns
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == ('object' or 'category' or 'bool')]
numerical_cols = [col for col in X_train.columns if col not in categorical_cols] # X_train[col].dtype != 'object'
high_cardinality_cols = [col for col in categorical_cols if X_train[col].nunique() > 500]
low_cardinality_cols = [col for col in categorical_cols if X_train[col].nunique() <= 500]


# Preprocessing pipeline for XGBoost
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Fit target encoder on training data only
target_encoder = TargetEncoder(cols=high_cardinality_cols)


print(f"Preprocessing pipeline for XGBoost:")
# Define a column transformer to handle categorical and numerical data
preprocessor_xgb = ColumnTransformer(
    transformers=[
        # ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numerical_cols),  # Impute and scale numerical data
        # # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),  # One-hot encode categorical data, sparse=False,
        # ('cat_low', OneHotEncoder(handle_unknown='ignore'), low_cardinality_cols),
        # ('cat_high', TargetEncoder(), high_cardinality_cols)
        # ('num', numerical_pipeline, numerical_cols),
        # ('cat_low', OneHotEncoder(handle_unknown='ignore'), low_cardinality_cols),
        # ('cat_high', 'passthrough', high_cardinality_cols)
        ('num', numerical_pipeline, numerical_cols),
        ('cat_low', OneHotEncoder(handle_unknown='ignore'), low_cardinality_cols),
        ('cat_high', 'passthrough', high_cardinality_cols)
    ]
)

print(f"Preprocessing pipeline for CatBoost:")
# Preprocessing pipeline for CatBoost
preprocessor_cb = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_cols),
        ('cat', 'passthrough', categorical_cols)  # Ensure categorical columns are included
    ]
)

## Calculate Scale Pos Weight
positive_class_count = y_train.value_counts().get(1, 0)
negative_class_count = y_train.value_counts().get(0, 0)
scale_pos_weight = negative_class_count / positive_class_count if positive_class_count > 0 else 1

# Applying the preprocessing and fitting XGBoost
X_train_split[high_cardinality_cols] = target_encoder.fit_transform(X_train_split[high_cardinality_cols], y_train_split)
X_valid_split[high_cardinality_cols] = target_encoder.transform(X_valid_split[high_cardinality_cols])


print(f"Processing pipeline for XGBoost:")
# XGBoost Pipeline
pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor_xgb),
    ('classifier', xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=random_state,
        eval_metric='auc'
    ))
])

print(f"Pipeline for XGBoost:")
# Fit XGBoost
pipeline_xgb.fit(X_train_split, y_train_split)

# Run garbage collection
gc.collect()


# # Ensure that all categorical features are of type string
# for col in categorical_cols:
#     X_train_split[col] = X_train_split[col].astype(str)
#     X_train_split[col] = X_train_split[col].fillna('missing_value').astype(str)  # Handle NaNs
# # Verify the changes
# print(X_train_split[categorical_cols].dtypes)
# # # Identify categorical feature indices for CatBoost
# categorical_features_indices = [X_train_split.columns.get_loc(col) for col in categorical_cols]
# # Ensure that the categorical features are defined correctly
# print(f"Categorical feature indices: {categorical_features_indices}")

# Ensure that all categorical features are of type string and handle NaNs
# for col in categorical_cols:
#     # Convert to string and handle NaN values
#     X_train_split[col] = X_train_split[col].fillna('missing_value').astype(str)
# # Check the first few rows to verify the conversion
# print(X_train_split[categorical_cols].head())

# # Alternatively, check if all values are strings using a simple check:
# for col in categorical_cols:
#     if not all(isinstance(val, str) for val in X_train_split[col].unique()):
#         print(f"Warning: Column {col} still contains non-string values")


for col in categorical_cols:  # cat_features is the list of your categorical columns
    if X_train_split[col].dtype == 'object':
        # Check if there are any numeric values
        mask = X_train_split[col].apply(lambda x: isinstance(x, float))
        if mask.any():
            print(f"Warning: Numeric values found in {col}. Converting to strings.")
            X_train_split[col] = X_train_split[col].astype(str)
    
# for col in numerical_cols:
#     # Convert to string and handle NaN values
#     X_train_split[col] = X_train_split[col].fillna('missing_value').astype(str)

# Verify the changes
# print(X_train_split[categorical_cols].dtypes)
# Identify categorical feature indices for CatBoost
categorical_features_indices = [X_train_split.columns.get_loc(col) for col in categorical_cols]
# Ensure that the categorical features are defined correctly
print(f"Categorical feature indices: {categorical_features_indices}")
# Check for any unexpected types in the training data before fitting
print(X_train_split.dtypes)



print(f"Processing pipeline for CatBoost:")
# CatBoost Pipeline
pipeline_cb = Pipeline(steps=[
    ('preprocessor', preprocessor_cb),
    ('classifier', CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        scale_pos_weight=scale_pos_weight,
        cat_features=categorical_features_indices,
        eval_metric='AUC',
        random_seed=random_state,
        early_stopping_rounds=50,
        verbose=100
    ))
])

print(f"Pipeline for CatBoost:")
# Fit CatBoost
pipeline_cb.fit(X_train_split, y_train_split)
# Run garbage collection
gc.collect()



print(f"Predict probabilities:")
# Predict probabilities
preds_xgb = pipeline_xgb.predict_proba(X_valid_split)[:, 1]
preds_cb = pipeline_cb.predict_proba(X_valid_split)[:, 1]
# Ensemble Predictions
ensemble_preds = (preds_xgb + preds_cb) / 2
# Evaluate

auc_xgb = roc_auc_score(y_valid_split, preds_xgb)
auc_cb = roc_auc_score(y_valid_split, preds_cb)
auc_ensemble = roc_auc_score(y_valid_split, ensemble_preds)

print(f"XGBoost ROC AUC: {auc_xgb:.4f}")
print(f"CatBoost ROC AUC: {auc_cb:.4f}")
print(f"Ensemble ROC AUC: {auc_ensemble:.4f}")

# # Define the parameter grid for XGBoost
# param_dist = {
#     'classifier__n_estimators': [50, 100, 200, 300, 400],        # Number of boosting rounds
#     'classifier__eta': [0.01, 0.05, 0.1, 0.2, 0.3],              # Learning rate
#     'classifier__gamma': [0, 0.1, 0.2, 0.5, 1],                  # Minimum loss reduction
#     'classifier__min_child_weight': [1, 2, 5, 10],               # Minimum sum of instance weight
#     'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],          # Subsampling of data for each tree
#     'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],   # Fraction of columns to sample for each tree
#     'classifier__max_depth': [3, 4, 5, 6, 8, 10]                 # Maximum depth of a tree
# }

# # Define the pipeline with the preprocessor and XGBoost model
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', xgb.XGBClassifier(
#         # n_estimators=100,
#         # max_depth=6,
#         # learning_rate=0.1,
#         # scale_pos_weight=scale_pos_weight,
#         # random_state=random_state

#         # random_state=random_state,
#         # use_label_encoder=False,
#         # verbosity=0
#         use_label_encoder=False, eval_metric='mlogloss', random_state=random_state
#     ))
# ])

# # Define Stratified K-Folds cross-validator
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# Initialize RandomizedSearchCV
# random_search = RandomizedSearchCV(
#     estimator=pipeline,
#     param_distributions=param_dist,
#     n_iter=150,
#     scoring='roc_auc',
#     cv=cv,
#     verbose=2,
#     random_state=random_state,
#     n_jobs=2
# )
# random_search = RandomizedSearchCV(
#     estimator=pipeline,
#     param_distributions=param_dist,
#     n_iter=50,            # Number of random combinations to try
#     scoring='roc_auc',    # Metric to evaluate
#     cv=5,                 # 5-fold cross-validation
#     random_state=random_state,     
#     verbose=1
# )

# gc.collect()

# # Fit with early stopping
# random_search.fit(
#     X_train_split,
#     y_train_split,
#     classifier__eval_set=[(X_valid_split, y_valid_split)],
#     classifier__early_stopping_rounds=50,
#     classifier__verbose=False
# )

# # # Fit RandomizedSearchCV
# # random_search.fit(X_train_split, y_train_split)

# gc.collect()

# # Best parameters
# print("Best Hyperparameters:")
# print(random_search.best_params_)

# # Best cross-validation AUC score
# print(f"Best CV AUC Score: {random_search.best_score_:.4f}")

# # Evaluate on the validation set
# best_model = random_search.best_estimator_
# y_valid_preds = best_model.predict_proba(X_valid_split)[:, 1]

# # Compute AUC score on validation data
# auc = roc_auc_score(y_valid_split, y_valid_preds)
# print(f"Validation ROC-AUC Score: {auc:.4f}")

# Plot ROC Curve
# fpr, tpr, thresholds = roc_curve(y_valid_split, y_valid_preds)
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()

# # Predict on the evaluation set (test data)
# y_preds = best_model.predict_proba(test_data.drop(columns=["id"]))[:, 1]

# # Make the submission file
# submission_df = pd.DataFrame({"id": test_data["id"],"Label": y_preds})
# submission_df["id"] = submission_df["id"].astype(int)
# submission_df.to_csv("random_search_best_model.csv", sep=",", index=False)
# print("Submission file 'random_search_best_model.csv' created successfully.")


print(f"Preparing submission file:")
# Step 2.1: Prepare Test Features
# Drop the 'id' column as it's not a feature
X_test = test_data.drop(columns=["id"])
# Step 2.2: Generate Predictions with XGBoost
y_preds_xgb = pipeline_xgb.predict_proba(X_test)[:, 1]
# Step 2.3: Generate Predictions with CatBoost
y_preds_cb = pipeline_cb.predict_proba(X_test)[:, 1]
# Step 2.4: Ensemble the Predictions by Averaging
y_preds_ensemble = (y_preds_xgb + y_preds_cb) / 2

# Step 2.5: Create the Submission DataFrame
submission_df = pd.DataFrame({"id": test_data["id"],"Label": y_preds_ensemble})
submission_df["id"] = submission_df["id"].astype(int)
# Step 2.6: Save the Submission File
submission_df.to_csv("ensemble_model_submission.csv", sep=",", index=False)
print("Submission file 'ensemble_model_submission.csv' created successfully.")

# Run garbage collection
gc.collect()


# End timer and print elapsed time
elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)
print(f"Ensemble model completed in {hours} hours {minutes} minutes {seconds} seconds.")

# # Fit the model
# pipeline.fit(X_train_split, y_train_split)

# # # Evaluate on the validation set
# y_valid_preds = pipeline.predict_proba(X_valid_split)[:, 1]

# ######################################################
# ## AUC-ROC
# ######################################################

# # Compute AUC score on validation data
# auc = roc_auc_score(y_valid_split, y_valid_preds)
# print(f"Validation AUC Score: {auc:.4f}")

# # Optional: Plot ROC Curve
# fpr, tpr, thresholds = roc_curve(y_valid_split, y_valid_preds)
# plt.figure()
# plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# ######################################################
# ## file
# ######################################################

# # Predict on the evaluation set (test data)
# y_preds = pipeline.predict_proba(test_data.drop(columns=["id"]))[:, 1]

# # Make the submission file
# submission_df = pd.DataFrame({"id": test_data["id"], "Label": y_preds})
# submission_df["id"] = submission_df["id"].astype(int)
# submission_df.to_csv("basic_model.csv", sep=",", index=False)

# # Run garbage collection
# gc.collect()

# ######################################################
# ## timer end
# ######################################################

# # End timer
# # Calculate the elapsed time in hours, minutes, and seconds
# elapsed_time = time.time() - start_time
# hours = int(elapsed_time // 3600)
# minutes = int((elapsed_time % 3600) // 60)
# seconds = int(elapsed_time % 60)

# # Print the elapsed time
# print("Process finished --- %d hours %d minutes %d seconds --- " % (hours, minutes, seconds))

