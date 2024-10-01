# 0.8573 / 8 hours 4 minutes 26 seconds
import gc
import ast
import time
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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
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

# Sample data
train_data = train_data.sample(frac=1, random_state=2345)

# Load the test data
test_data = pd.read_csv("ctr_test.csv")

#
random_state = 2345

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
for df in tqdm([train_data, test_data], desc="Creating Good Hours Feature"):
    df['good_hours'] = df['auction_hour'].apply(lambda x: 1 if x in good_hours else 0)

##################### lists

# for col in ['action_list_0', 'action_list_1', 'action_list_2']:
#     train_data[f'{col}_count'] = train_data[col].apply(lambda x: len(str(x).split(',')))
#     test_data[f'{col}_count'] = test_data[col].apply(lambda x: len(str(x).split(',')))

# for col in ['action_list_0', 'action_list_1', 'action_list_2']:
#     train_data[f'{col}_unique_count'] = train_data[col].apply(lambda x: len(set(str(x).split(','))))
#     test_data[f'{col}_unique_count'] = test_data[col].apply(lambda x: len(set(str(x).split(','))))

##################################################################################
# # Convert string representation of lists into actual lists
# for df in tqdm([train_data, test_data], desc="Converting Auction Lists"):
#     df['auction_list_0'] = df['auction_list_0'].apply(
#         lambda x: ast.literal_eval(x) if isinstance(x, str) else []  # Return empty list if NaN or None
#     )

# # One-hot encoding for auction_list_0
# mlb = MultiLabelBinarizer()
# # Create a dictionary that maps each unique category to a unique token
# mlb.fit(train_data['auction_list_0'])
# unique_auction_tokens = {category: idx for idx, category in enumerate(mlb.classes_)}
# # Apply the mapping to convert lists into token lists with progress tracking
# for df in tqdm([train_data, test_data], desc="Tokenizing Auction Lists"):
#     df['auction_list_0_tokenized'] = df['auction_list_0'].apply(
#         lambda lst: [unique_auction_tokens.get(item, -1) for item in lst] #-1 for unknown
#     )

# ###

# # Convert string representation of action lists into actual lists
# for col in ['action_list_1', 'action_list_2']:
#     for df in tqdm([train_data, test_data], desc=f"Processing {col} Lists"):
#         df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else [] if pd.isna(x) else x)


# # Create tokenized versions of action_list_1 and action_list_2
# for col in ['action_list_1', 'action_list_2']:
#     for df in tqdm([train_data, test_data], desc=f"Tokenizing {col}"):
#         df[f'{col}_tokenized'] = df[col].apply(lambda lst: [abs(int(item)) for item in lst] if isinstance(lst, list) else [])


# # Calculate mean and max for action_list_1 and action_list_2
# for col in ['action_list_1', 'action_list_2']:
#     for df in tqdm([train_data, test_data], desc=f"Calculating {col} Mean and Max"):
#         # Calculate mean
#         df[f'{col}_mean'] = df[col].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)
#         # Calculate max
#         df[f'{col}_max'] = df[col].apply(lambda x: max(x) if len(x) > 0 else 0)

# # Calculate overall mean and max from training data for test data
# for col in ['action_list_1', 'action_list_2']:
#     overall_mean = train_data[f'{col}_mean'].mean()
#     overall_max = train_data[f'{col}_max'].max()

#     for df in tqdm([train_data, test_data], desc=f"Applying overall {col} Mean and Max"):
#         df[f'{col}_mean'] = df[f'{col}_mean'].apply(lambda x: overall_mean if x == 0 else x)
#         df[f'{col}_max'] = df[f'{col}_max'].apply(lambda x: overall_max if x == 0 else x)

####

# # Transform the multi-label columns
# # for df in tqdm([train_data, test_data], desc=f"Transforming the multi-label column: {col}"):
# # Apply MultiLabelBinarizer to multi-label columns
# mlb_action_1 = MultiLabelBinarizer()
# mlb_action_2 = MultiLabelBinarizer()

# # Fit and transform multi-label columns
# action_list_1_transformed = mlb_action_1.fit_transform(train_data['action_list_1'])
# action_list_2_transformed = mlb_action_2.fit_transform(train_data['action_list_2'])

# # Create DataFrames for the transformed data
# action_list_1_df = pd.DataFrame(action_list_1_transformed, columns=[f'action_1_{i}' for i in range(action_list_1_transformed.shape[1])])
# action_list_2_df = pd.DataFrame(action_list_2_transformed, columns=[f'action_2_{i}' for i in range(action_list_2_transformed.shape[1])])

# # Concatenate back to original DataFrame
# train_data = pd.concat([train_data.reset_index(drop=True), action_list_1_df, action_list_2_df], axis=1)
# train_data.drop(columns=['action_list_1', 'action_list_2'], inplace=True)  # Drop original columns if needed

   
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
    df['rolling_bidfloor'] = df.groupby('device_id')['auction_bidfloor'].transform(lambda x: x.rolling(window=5).mean())

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

# Dropping columns with high cardinality 
# Function to determine if a column is unhashable
def is_unhashable(column):
    try:
        # Try converting to a set to check for uniqueness
        _ = set(column)
        return False
    except TypeError:
        return True
# Identify high cardinality columns, excluding unhashable types
high_cardinality_cols = []
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        if is_unhashable(train_data[col]):
            print(f"Skipping column {col} due to unhashable type.")
            continue
        if train_data[col].nunique() > 500:  
            high_cardinality_cols.append(col)
# Drop high cardinality columns from both train and test data
train_data.drop(high_cardinality_cols, axis=1, inplace=True)
test_data.drop(high_cardinality_cols, axis=1, inplace=True)
print(f"Dropped columns high cardinality: {high_cardinality_cols}")

######################################################
## Model
######################################################

# Train a tree on the train data (sampling training data)
# train_data = train_data.sample(frac=8/10)
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])


## Calculate Scale Pos Weight
positive_class_count = y_train.value_counts().get(1, 0)
negative_class_count = y_train.value_counts().get(0, 0)
scale_pos_weight = negative_class_count / positive_class_count if positive_class_count > 0 else 1

# Train/test split for validation
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.15, random_state=random_state)


# Columns that contain lists (multi-categories)
# list_cols = [col for col in X_train.columns if isinstance(X_train[col].iloc[0], list)]
# Separate the categorical and numerical columns
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
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),  # One-hot encode categorical data, sparse=False,
        # ('list', make_pipeline(list_transformer), list_cols)  # Use MultiLabelBinarizer for list-like columns
    ]
)

# # Define the pipeline with the preprocessor and XGBoost model
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', xgb.XGBClassifier(
#         n_estimators=100,
#         max_depth=6,
#         learning_rate=0.1,
#         scale_pos_weight=scale_pos_weight,
#         random_state=random_state
#     ))
# ])

# # Fit the model
# pipeline.fit(X_train_split, y_train_split)

# # # Evaluate on the validation set
# y_valid_preds = pipeline.predict_proba(X_valid_split)[:, 1]

g1 = {'colsample_bytree': 0.85, 'gamma': 2.12, 'learning_rate': 0.01, 'max_depth': 7,
          'min_child_weight': 14, 'n_estimators': 2500, 'reg_lambda': 0.2, 'subsample': 0.85}

g2 = {'colsample_bytree': 0.6509900925068455, 'gamma': 1.1805850841291976, 'learning_rate': 0.009732823220209536,
          'max_depth': 8, 'min_child_weight': 63.01501268012936, 'n_estimators': 2800, 'reg_lambda': 1.2219990638829437,
          'subsample': 0.9692112596805615}
g3 = {'colsample_bytree': 0.5703515463135813, 'gamma': 4.505167386980914, 'learning_rate': 0.0107422753705186,
          'max_depth': 8, 'min_child_weight': 16.198380840241803, 'n_estimators': 2588, 'reg_lambda': 9.777449149282896,
          'subsample': 0.6621830744247428}
g4 = {'colsample_bytree': 0.6138573660646992, 'gamma': 4.515652849784547, 'learning_rate': 0.012317873295396919,
          'max_depth': 10, 'min_child_weight': 31.864967919632974, 'n_estimators': 2784, 'reg_lambda': 9.94406970874546,
          'subsample': 0.6914959499656252} # This worked even though parameters are from xgboost, those who don't apply to LGBM are ignored.

g5 = {'boosting_type': 'gbdt', 'colsample_bytree': 0.7298850189031411, 'learning_rate': 0.011223985889567765,
          'min_child_samples': 75, 'n_estimators': 1700, 'num_leaves': 77, 'reg_alpha': 2.0453339399606656, 'reg_lambda': 1.1015957272258972,
          'subsample': 0.6839425708957405}

g6 = {'bagging_temperature': 0.04980344808007908, 'bootstrap_type': 'Bayesian', 'border_count': 254,
          'depth': 11, 'iterations': 992, 'l2_leaf_reg': 2.08325705091333, 'learning_rate': 0.020415818828056046,
          'od_type': 'Iter', 'od_wait': 10, 'random_strength': 3.2037112912872576}

g7 = {'bagging_temperature': 0.6554656944997739, 'bootstrap_type': 'Bayesian', 'border_count': 254, 'depth': 9,
         'iterations': 2359, 'l2_leaf_reg': 77.77538901843211, 'learning_rate': 0.0282821275240532,
         'od_type': 'IncToDec', 'od_wait': 28, 'random_strength': 3.369002203843343, 'verbose': False}

model1 = xgb.XGBClassifier(objective='binary:logistic', seed=random_state, eval_metric='auc', **g1, tree_method='hist')
model2 = xgb.XGBClassifier(objective='binary:logistic', seed=random_state, eval_metric='auc', **g2, tree_method='hist')
model3 = xgb.XGBClassifier(objective='binary:logistic', seed=random_state, eval_metric='auc', **g3, tree_method='hist')

model4 = lgb.LGBMClassifier(objective='binary', seed=random_state, **g4, device='cpu')
model5 = lgb.LGBMClassifier(objective='binary', seed=random_state, **g5, device='cpu')

model6 = CatBoostClassifier(random_seed=random_state, eval_metric='AUC', task_type="CPU", **g6)
model7 = CatBoostClassifier(random_seed=random_state, eval_metric='AUC', task_type="CPU", **g7)

models = [model1, model2, model3, model4, model5, model6, model7]

gc.collect()

# Loop through the models, build pipelines, fit, and evaluate
for i, model in enumerate(models):
    print(f"Training model {i+1}")
    
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
    print(f"Model {i+1} AUC: {auc_score:.4f}")

    gc.collect()

print("Train", train_data.columns)
print('Test', test_data.columns)

######################################################
## AUC-ROC
######################################################

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

######################################################
## file
######################################################

# # Predict on the evaluation set (test data)
# y_preds = pipeline.predict_proba(test_data.drop(columns=["id"]))[:, 1]

# # Make the submission file
# submission_df = pd.DataFrame({"id": test_data["id"], "Label": y_preds})
# submission_df["id"] = submission_df["id"].astype(int)
# submission_df.to_csv("basic_model.csv", sep=",", index=False)

# Predict on the evaluation set (test data)
# Create an empty DataFrame to store predictions for submission
submission_preds = []
for i, model in enumerate(models):
    print(f"Generating predictions for model {i+1}")
    # Define the pipeline for prediction
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    # Fit the pipeline on the entire training data
    pipeline.fit(X_train, y_train)  # Ensure that you use the full training data for final predictions
    # Predict on the test data
    y_preds = pipeline.predict_proba(test_data.drop(columns=["id"]))[:, 1]
    # Add predictions to the list
    submission_preds.append(y_preds)

# If you want an ensemble prediction, average them (optional)
final_preds = sum(submission_preds) / len(submission_preds)

# Make the submission file
submission_df = pd.DataFrame({"id": test_data["id"], "Label": final_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)


######################################################
## garbage collector
######################################################
# Clean up memory and run garbage collection before ending
del pipeline, X_train_split, X_valid_split, y_train_split, y_valid_split, y_valid_preds, y_preds, submission_df
gc.collect()

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

