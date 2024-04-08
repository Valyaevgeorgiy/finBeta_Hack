import numpy as np
import pandas as pan
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

train_df = pan.read_parquet("data/train_data.pqt")
test_df = pan.read_parquet("..data/test_data.pqt")