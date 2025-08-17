# dt-lgb-comparison-mit

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Decision Tree と LightGBM を比較してみるサンプルリポジトリです。  
サンプルデータは車の好みデータ（架空）を使用しています。

精度安定版追加（2025/8/17)
tree_s_l_f5.py
- 単一分割(tree_s_l.py)の評価は 分割方法に依存 しやすく、評価が安定しない
- 5-fold クロスバリデーションを使うと、より安定した推定値 が得られる
- 標準偏差からはモデルの 精度の安定性 も確認できる

---

## 🎯 対象者

- 決定木（Decision Tree）と LightGBM の性能差を比較したい方
- 精度評価（Accuracy, F1スコア）や混同行列を用いてモデルの性能を確認したい方
- LightGBMのラベル0スタートの重要性やカテゴリ変数の扱いを学びたい方

---

## ⚙️ 機能

- 決定木（Decision Tree）と LightGBM の性能差を確認
- 精度評価（Accuracy, F1スコア）や混同行列で比較
- LightGBMのラベル0スタートの重要性やカテゴリ変数の扱いを学ぶ

---

## 🛠 動作環境

- Python 3.x
- 必要なライブラリ：
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - lightgbm

必要なパッケージは以下のコマンドでインストールしてください。
```
pip install numpy pandas matplotlib scikit-learn lightgbm
```

# 📁 フォルダ構成
```

├─ 1_flow/
│ │─ tree_s_l.py                # 実行スクリプト
│ └─ tree_s_l_f5.py             # 5-fold実行スクリプト
├─ 2_data/
│ └─ sample_car_data.csv       # サンプルデータ
├─ 3_output/                   # 決定木出力画像用（自動作成）

```

# 📄 入力データフォーマット例
- 2_data/sample_data.csv を参照


# 🚀 使い方
1. 2_data/sample_data.csv にサンプルデータを準備
2. 以下のコマンドで実行
```
python 1_flow/tree_s_l.py
```
3. 結果や可視化画像は 3_output/ に出力されます

# ⚠️ 特記事項
- LightGBM の多クラス分類では、目的変数ラベルは 0スタートの連続整数 必須
- 説明変数は数値そのまま or category 型で渡す
- 決定木と LightGBM の木は構造が異なるため、可視化結果の見え方も異なる
- 決定木：1本で分類
- LightGBM：多数の木を組み合わせて予測

# 📊 精度評価
- Accuracy（正解率）
- F1スコア（加重平均、クラス不均衡対応）
- 混同行列で誤分類傾向を確認可能

# 🤝 貢献方法
プロジェクトへの貢献は以下の方法で歓迎します：
- バグ報告や機能追加の提案は Issues を通じて行ってください
- コードの改善や新機能の追加は Pull Request を作成してください
- ドキュメントの改善や翻訳も歓迎します

# 📄 LICENSE
MIT License（詳細はLICENSEファイルをご参照ください）

#### 開発者： iwakazusuwa(Swatchp)

