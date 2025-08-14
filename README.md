<img width="254" height="26" alt="image" src="https://github.com/user-attachments/assets/2b6cb17b-f254-4356-8d3f-557b12466c1c" /># dt-lgb-comparison-mit

Decision Tree と LightGBM を比較してみるサンプルリポジトリです。  
サンプルデータは車の好みデータ（架空）を使用しています。

---

# 機能

- 決定木（Decision Tree）と LightGBM の性能差を確認する
- 精度評価（Accuracy, F1スコア）や混同行列で比較
- LightGBMのラベル0スタートの重要性やカテゴリ変数の扱いを学ぶ

---

# 動作環境

- Python 3.x
- ライブラリ
  - numpy, pandas, matplotlib
  - scikit-learn
  - lightgbm

必要なパッケージは以下のコマンドでインストールしてください。
```
pip install numpy pandas matplotlib scikit-learn lightgbm
```

## フォルダ構成
```
<img width="254" height="26" alt="image" src="https://github.com/user-attachments/assets/e1e3e3de-2e96-483c-9d97-ad9c45f7c5bf" />

├─ 1_flow/
│ └─ tree_s_l.py               # 実行スクリプト
├─ 2_data/
│ └─ sample_car_data.csv       # サンプルデータ
├─ 3_output/                   # 決定木出力画像用（自動作成）

```

# 入力データフォーマット例
2_data/sample.csv を参照ください。

# 使い方
このスクリプトは、コマンドライン操作なしで、ファイルをダブルクリックするだけで実行できます。

1. スクリプト実行
2. 3_output/ に決定木や LightGBM の可視化画像が保存されます
3. コンソールに Accuracy, F1 スコアが表示されます

# 特記事項
- LightGBM の多クラス分類では、目的変数ラベルは 0スタートの連続整数 必須
- 説明変数は数値そのまま or category 型で渡す
- 決定木と LightGBM の木は構造が異なるため、可視化結果の見え方も異なる
- 決定木：1本で分類
- LightGBM：多数の木を組み合わせて予測

# 精度評価
- Accuracy（正解率）
- F1スコア（加重平均、クラス不均衡対応）
- 混同行列で誤分類傾向を確認可能

## LICENSE
MIT License（詳細はLICENSEファイルをご参照ください）

#### 開発者： iwakazusuwa(Swatchp)
<img width="254" height="101" alt="image" src="https://github.com/user-attachments/assets/1d122d0a-472f-4d39-a3a2-b10a3ac1148d" />
