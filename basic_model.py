import gc
import time
import pandas as pd
import xgboost as xgb

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# start timer 
start_time = time.time()

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# List all the file names
file_names = ["ctr_15.csv", "ctr_16.csv", "ctr_17.csv", "ctr_18.csv", "ctr_19.csv", "ctr_20.csv", "ctr_21.csv"]

# Read and concatenate all the CSV files
train_data = pd.concat([pd.read_csv(file) for file in file_names])

# Sample 1/10th of each dataset and concatenate
# sampled_data = []
# for file in file_names:
#     data = pd.read_csv(file)
#     sampled_data.append(data.sample(frac=1/10))

# Combine the sampled datasets into one DataFrame
# train_data = pd.concat(sampled_data)

# Load the test data
eval_data = pd.read_csv("ctr_test.csv")

# Feature Engineering: Creating new features from auction_time
train_data['auction_hour'] = pd.to_datetime(train_data['auction_time'], unit='s').dt.hour
train_data['auction_day'] = pd.to_datetime(train_data['auction_time'], unit='s').dt.dayofweek

# Train a tree on the train data (sampling training data)
train_data = train_data.sample(frac=1/10)
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])

# Separate the categorical and numerical columns
# categorical_cols = [col for col in X_train.columns if 'categorical' in col]
# numerical_cols = [col for col in X_train.columns if col not in categorical_cols]
categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
numerical_cols = [col for col in X_train.columns if X_train[col].dtype != 'object']

# Define a column transformer to handle categorical and numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numerical_cols),  # Impute and scale numerical data
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # One-hot encode categorical data
    ]
)


# Define the pipeline with the preprocessor and XGBoost model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=2345))
])

# Train/test split for validation
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.2, random_state=2345)

# Fit the model
pipeline.fit(X_train_split, y_train_split)

# Evaluate on the validation set
y_valid_preds = pipeline.predict_proba(X_valid_split)[:, 1]

# Predict on the evaluation set (test data)
eval_data['auction_hour'] = pd.to_datetime(eval_data['auction_time'], unit='s').dt.hour
eval_data['auction_day'] = pd.to_datetime(eval_data['auction_time'], unit='s').dt.dayofweek
y_preds = pipeline.predict_proba(eval_data.drop(columns=["id"]))[:, 1]


# X_train = X_train.select_dtypes(include='number')
# del train_data
# gc.collect()

# cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=2345))
# cls.fit(X_train, y_train)

# # Predict on the evaluation set
# eval_data = eval_data.select_dtypes(include='number')
# y_preds = cls.predict_proba(eval_data.drop(columns=["id"]))[:, cls.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)


# end timer
# Calculate the elapsed time in hours, minutes, and seconds
elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

# Print the elapsed time
print("Process finished --- %d hours %d minutes %d seconds --- " % (hours, minutes, seconds))