import pandas as pd
import numpy as np
xb = pd.read_csv("basic_model_xgboost.csv")
lgbm = pd.read_csv("basic_model_lgbm2_full.csv")


# If you want an ensemble prediction, average them (optional)
final_preds = (xb['Label'] + lgbm['Label']) / 2

# Make the submission file
submission_df = pd.DataFrame({"id": range(1, len(final_preds) + 1), "Label": final_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("ensemble_model_lgbm_xgboost.csv", sep=",", index=False)
    

