import gc
import time
import pandas as pd
import xgboost as xgb

from tqdm import tqdm
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# Read and concatenate all the CSV files
train_data = pd.concat([pd.read_csv(file) for file in file_names])

# Load the test data
test_data = pd.read_csv("ctr_test.csv")

# Feature Engineering: Creating new features from auction_time
# train_data['auction_hour'] = pd.to_datetime(train_data['auction_time'], unit='s').dt.hour
# train_data['auction_day'] = pd.to_datetime(train_data['auction_time'], unit='s').dt.dayofweek

##########################################################################################################################
##  FEATURE ENGINEERING
##########################################################################################################################


# Convert auction_time (Unix time) to datetime
train_data['auction_time'] = pd.to_datetime(train_data['auction_time'], unit='s')
test_data['auction_time'] = pd.to_datetime(test_data['auction_time'], unit='s')

# Create new features from auction_time
for df in [train_data, test_data]:
    df['auction_hour'] = df['auction_time'].dt.hour             # Hour of the auction
    df['auction_minute'] = df['auction_time'].dt.minute         # Minute of the auction
    df['auction_day_of_week'] = df['auction_time'].dt.dayofweek # Day of the week (Monday=0, Sunday=6)
    df['auction_day'] = df['auction_time'].dt.day               # Day of the month
    df['auction_month'] = df['auction_time'].dt.month           # Month of the year
    df['auction_year'] = df['auction_time'].dt.year             # Year of the auction
    df['is_weekend'] = df['auction_day_of_week'].apply(lambda x: 1 if x >= 5 else 0) #binary features for weekends

##################### good_hours

# Step 1: Identify the most popular hours (good hours)
auction_hour_counts = train_data['auction_hour'].value_counts().sort_values(ascending=False)
# Step 2: Define good hours (e.g., top 25% most frequent hours)
top_25_percent = int(0.25 * len(auction_hour_counts))
good_hours = auction_hour_counts.index[:top_25_percent].tolist()
# Step 3: Create the 'good_hours' feature
for df in [train_data, test_data]:
    df['good_hours'] = df['auction_hour'].apply(lambda x: 1 if x in good_hours else 0)
    

# Now, dropping the original 'auction_time' since it's not needed anymore for the model
train_data = train_data.drop(columns=['auction_time'])
test_data = test_data.drop(columns=['auction_time'])


##################### Reducing size 
# Group rare categories
threshold = 100  # You can adjust this threshold
value_counts = train_data['device_id_type'].value_counts()
rare_categories = value_counts[value_counts < threshold].index
train_data['device_id_type'] = train_data['device_id_type'].replace(rare_categories, 'Other')



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
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, y_train, test_size=0.3, random_state=2345)

# Separate the categorical and numerical columns
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
    ('classifier', xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=2345
    ))
])


# Fit the model
pipeline.fit(X_train_split, y_train_split)

# # Evaluate on the validation set
y_valid_preds = pipeline.predict_proba(X_valid_split)[:, 1]

# Compute AUC score on validation data
auc = roc_auc_score(y_valid_split, y_valid_preds)
print(f"Validation AUC Score: {auc:.4f}")

# Optional: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_valid_split, y_valid_preds)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()



# Predict on the evaluation set (test data)
# test_data['auction_hour'] = pd.to_datetime(test_data['auction_time'], unit='s').dt.hour
# test_data['auction_day'] = pd.to_datetime(test_data['auction_time'], unit='s').dt.dayofweek


# Predict on the evaluation set (test data)
y_preds = pipeline.predict_proba(test_data.drop(columns=["id"]))[:, 1]


# Make the submission file
submission_df = pd.DataFrame({"id": test_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)


# End timer
# Calculate the elapsed time in hours, minutes, and seconds
elapsed_time = time.time() - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

# Print the elapsed time
print("Process finished --- %d hours %d minutes %d seconds --- " % (hours, minutes, seconds))