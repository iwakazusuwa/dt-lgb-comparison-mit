
# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
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
save_matrix_path  = os.path.join(output_path, "Confusion_Matrix.png")

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
# データ分割（test_size=0.25）
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_enc, test_size=0.25, random_state=0, stratify=y_enc
)
print("訓練データ数:", len(X_train))
print("テストデータ数:", len(X_test))

# =========================
# 決定木モデル（訓練データのみで学習）
# =========================
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

# =========================
# 決定木の可視化 → 全データで学習
# =========================
dt_model_all = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model_all.fit(X_df, y_enc) 

plt.figure(figsize=(20,10))
plot_tree(dt_model, filled=True, feature_names=X_df.columns, class_names=class_names)
plt.title("scikit-learn Decision Tree")
plt.savefig(save_tree_path, dpi=300)
# plt.show()

# =========================
# LightGBMモデル（訓練データのみで学習）
# =========================
objective = 'binary' if len(class_names) == 2 else 'multiclass'
metric = 'binary_error' if objective == 'binary' else 'multi_error'
params = {'objective': objective, 'metric': metric, 'verbose': -1}
if objective == 'multiclass':
    params['num_class'] = len(class_names)

lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=X_df.columns.tolist())
lgb_model = lgb.train(params, lgb_train, num_boost_round=50)

y_pred_lgb_prob = lgb_model.predict(X_test)
y_pred_lgb = (y_pred_lgb_prob > 0.5).astype(int) if objective=='binary' else np.argmax(y_pred_lgb_prob, axis=1)

# =========================
# LightGBM（全データで学習、可視化用）
# =========================
lgb_all = lgb.Dataset(X_df, label=y_enc, feature_name=X_df.columns.tolist())
lgb_model_all = lgb.train(params, lgb_all, num_boost_round=50)

ax_all = lgb.plot_tree(
    lgb_model_all, tree_index=0, figsize=(20, 10),
    show_info=['split_gain', 'internal_value', 'leaf_count']
)
plt.title("LightGBM Tree (All Data)")
plt.savefig(save_lgb_path, dpi=300)
# plt.show()

# =========================
# 精度評価
# =========================
print("【DecisionTree】 Accuracy:", accuracy_score(y_test, y_pred_dt))
print("【DecisionTree】 F1 Score:", f1_score(y_test, y_pred_dt, average='weighted'))

print("【LightGBM】     Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("【LightGBM】     F1 Score:", f1_score(y_test, y_pred_lgb, average='weighted'))

# =========================
# 混同行列
# =========================
fig, ax = plt.subplots(1, 2, figsize=(12,5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, ax=ax[0])
ax[0].set_title("scikit-learn Tree Confusion Matrix")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lgb, ax=ax[1])
ax[1].set_title("LightGBM Confusion Matrix")
plt.tight_layout()
plt.savefig(save_matrix_path, dpi=300)
plt.show()

