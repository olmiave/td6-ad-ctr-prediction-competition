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
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])

# Train/test split for validation
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.15, random_state=random_state, stratify=y_train)

# Separate the categorical and numerical columns
categorical_cols = [col for col in X_train.columns if X_train[col].dtype in ['object', 'category', 'bool']]
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

print(f"Finished Pipeline for XGBoost:")

# Run garbage collection
gc.collect()


# for col in X_train_split:  # cat_features is the list of your categorical columns
#     if X_train_split[col].dtype == 'object':
#         # Check if there are any numeric values
#         mask = X_train_split[col].apply(lambda x: isinstance(x, float))
#         if mask.any():
#             print(f"Warning: Numeric values found in {col}. Converting to strings.")
#             X_train_split[col] = X_train_split[col].astype(str)

for col in categorical_cols:
    if X_train_split[col].dtype not in [object, 'category']:
        print(f"Converting {col} to string to ensure compatibility with CatBoost.")
        X_train_split[col] = X_train_split[col].astype(str)
        X_valid_split[col] = X_valid_split[col].astype(str)
        test_data[col] = test_data[col].astype(str)
    else:
        # Check for float values in categorical columns
        if X_train_split[col].dtype == 'object':
            mask = X_train_split[col].apply(lambda x: isinstance(x, float))
            if mask.any():
                print(f"Warning: Numeric values found in {col}. Converting to strings.")
                X_train_split[col] = X_train_split[col].astype(str)
                X_valid_split[col] = X_valid_split[col].astype(str)
                test_data[col] = test_data[col].astype(str)

for col in categorical_cols:
    non_string_values = X_train_split[col].apply(lambda x: isinstance(x, (float, int)) and not isinstance(x, bool))
    if non_string_values.any():
        print(f"Non-string values found in column {col}")
        # Fill or drop NaN/NA values to avoid the masking error
        non_string_values_filled = non_string_values.fillna(False)  # Convert NA/NaN to False for boolean mask
        print(X_train_split[non_string_values_filled][col].unique())

for col in categorical_cols:
    non_string_values = X_train_split[col].apply(lambda x: not isinstance(x, str)).sum()
    if non_string_values > 0:
        print(f"Conversion failed for column {col}.")
    else:
        print(f"All values in {col} are now strings.")




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

print(f"Finished Pipeline for CatBoost:")

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