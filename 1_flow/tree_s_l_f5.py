# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgb

# =========================
# ディレクトリ・ファイル設定
# =========================
INPUT_FOLDER = '2_data'
INPUT_FILE = 'sample_car_data.csv'
OUTPUT_FOLDER = '3_output'

ID = 'customer_id'
target_col = 'manufacturer'
numeric_cols = ["family", "age", "children", "income"]

current_path = os.getcwd()
parent_path = os.path.dirname(current_path)
input_path = os.path.join(parent_path, INPUT_FOLDER, INPUT_FILE)
output_path = os.path.join(parent_path, OUTPUT_FOLDER)
os.makedirs(output_path, exist_ok=True)

save_tree_path = os.path.join(output_path, "DecisionTree.png")
save_lgb_path  = os.path.join(output_path, "LightGBM_Tree.png")
save_matrix_path = os.path.join(output_path, "Confusion_Matrix_Avg.png")

# =========================
# CSV読み込み
# =========================
try:
    df = pd.read_csv(input_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(input_path, encoding="cp932")

# =========================
# カテゴリ列自動判定
# =========================
exclude_cols = [ID, target_col]
categorical_cols = [col for col in df.columns if col not in exclude_cols + numeric_cols]
df[categorical_cols] = df[categorical_cols].astype("category")

# =========================
# 説明変数と目的変数
# =========================
X_df = df.drop([ID, target_col], axis=1)
y_df = df[target_col]

# =========================
# ラベルを0始まりに変換（LightGBM対応）
# =========================
le = LabelEncoder()
y_enc = le.fit_transform(y_df)
class_names = [str(c) for c in le.classes_]

print("クラス:", class_names)

# =========================
# クロスバリデーション設定
# =========================
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

dt_acc_list, dt_f1_list = [], []
lgb_acc_list, lgb_f1_list = [], []

cm_dt_sum = np.zeros((len(class_names), len(class_names)), dtype=float)
cm_lgb_sum = np.zeros((len(class_names), len(class_names)), dtype=float)

objective = 'binary' if len(class_names) == 2 else 'multiclass'
metric = 'binary_error' if objective == 'binary' else 'multi_error'
params = {'objective': objective, 'metric': metric, 'verbose': -1}
if objective == 'multiclass':
    params['num_class'] = len(class_names)

for fold, (train_idx, test_idx) in enumerate(skf.split(X_df, y_enc), 1):
    X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]
    
    # --- 決定木 ---
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    dt_acc_list.append(accuracy_score(y_test, y_pred_dt))
    dt_f1_list.append(f1_score(y_test, y_pred_dt, average='weighted'))
    cm_dt_sum += confusion_matrix(y_test, y_pred_dt, labels=range(len(class_names)))
    
    # --- LightGBM ---
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=X_df.columns.tolist())
    lgb_model = lgb.train(params, lgb_train, num_boost_round=50)
    y_pred_lgb_prob = lgb_model.predict(X_test)
    y_pred_lgb = (y_pred_lgb_prob > 0.5).astype(int) if objective=='binary' else np.argmax(y_pred_lgb_prob, axis=1)
    lgb_acc_list.append(accuracy_score(y_test, y_pred_lgb))
    lgb_f1_list.append(f1_score(y_test, y_pred_lgb, average='weighted'))
    cm_lgb_sum += confusion_matrix(y_test, y_pred_lgb, labels=range(len(class_names)))


# =========================
# 可視化用モデル（全データ学習）
# =========================
dt_model_all = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model_all.fit(X_df, y_enc)
plt.figure(figsize=(20,10))
plot_tree(dt_model_all, filled=True, feature_names=X_df.columns, class_names=class_names)
plt.title("Decision Tree (All Data)")
plt.savefig(save_tree_path, dpi=300)

lgb_all = lgb.Dataset(X_df, label=y_enc, feature_name=X_df.columns.tolist())
lgb_model_all = lgb.train(params, lgb_all, num_boost_round=50)
ax_all = lgb.plot_tree(
    lgb_model_all, tree_index=0, figsize=(20, 10),
    show_info=['split_gain', 'internal_value', 'leaf_count']
)
plt.title("LightGBM Tree (All Data)")
plt.savefig(save_lgb_path, dpi=300)



# 平均混同行列
cm_dt_avg = cm_dt_sum / n_splits
cm_lgb_avg = cm_lgb_sum / n_splits

# =========================
# 精度表示
# =========================
print(f"【DecisionTree】 Accuracy: {np.mean(dt_acc_list):.3f} ± {np.std(dt_acc_list):.3f}")
print(f"【DecisionTree】 F1 Score: {np.mean(dt_f1_list):.3f} ± {np.std(dt_f1_list):.3f}")
print(f"【LightGBM】     Accuracy: {np.mean(lgb_acc_list):.3f} ± {np.std(lgb_acc_list):.3f}")
print(f"【LightGBM】     F1 Score: {np.mean(lgb_f1_list):.3f} ± {np.std(lgb_f1_list):.3f}")

# =========================
# 平均混同行列の可視化（from_predictions風）
# =========================
fig, ax = plt.subplots(1, 2, figsize=(12,5))

disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt_avg, display_labels=class_names)
disp_dt.plot(ax=ax[0], cmap=plt.cm.Blues, values_format=".2f")
ax[0].set_title("DecisionTree Avg Confusion Matrix")

disp_lgb = ConfusionMatrixDisplay(confusion_matrix=cm_lgb_avg, display_labels=class_names)
disp_lgb.plot(ax=ax[1], cmap=plt.cm.Blues, values_format=".2f")
ax[1].set_title("LightGBM Avg Confusion Matrix")

plt.tight_layout()
plt.savefig(save_matrix_path, dpi=300)
plt.show()

