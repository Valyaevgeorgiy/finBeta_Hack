import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import svm

train_df = pd.read_parquet("data/train_data.pqt")
test_df = pd.read_parquet("data/test_data.pqt")
ar=pd.DataFrame(train_df).head(3)
cat_cols = ["channel_code", "city", "city_type", "okved", "segment", "start_cluster", "index_city_code", "ogrn_month", "ogrn_year",]
train_df[cat_cols] = train_df[cat_cols].astype("category")
test_df[cat_cols] = test_df[cat_cols].astype("category")
X = train_df.drop(["id", "date", "end_cluster"], axis=1)
y = train_df["end_cluster"]
print(X)
print(y)
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LGBMClassifier(verbosity=-1, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)
def weighted_roc_auc(y_true, y_pred, labels, weights_dict):
    unnorm_weights = np.array([weights_dict[label] for label in labels])
    weights = unnorm_weights / unnorm_weights.sum()
    classes_roc_auc = roc_auc_score(y_true, y_pred, labels=labels,
                                    multi_class="ovr", average=None)
    return sum(weights * classes_roc_auc)
cluster_weights = pd.read_excel("data/cluster_weights.xlsx").set_index("cluster")
weights_dict = cluster_weights["unnorm_weight"].to_dict()
y_pred_proba = model.predict_proba(x_val)
print(y_pred_proba)

