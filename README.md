# ドメイン適応とは？
ドメイン適応（Domain Adaptation, DA）は、異なるデータ分布（ドメイン）間で学習したモデルの知識を適応させる技術です。  
一般的に、ソースドメイン（ラベル付きの学習データ）で学習したモデルは、異なるターゲットドメイン（未学習のデータ）での性能が低下します。  
ドメイン適応の手法を用いることで、この性能低下を抑え、ターゲットドメインに適応させることが可能です。

特に教師なしドメイン適応（UDA：Unsupervised Domain Adaptation）は、ラベル付きのソースドメインのデータを用いて学習したモデルを、ラベルのないターゲットドメインのデータに適応させる手法のことです。

---

## **クローズドセットドメイン適応（Closed-set Domain Adaptation）**
**定義:**  
- **ソースドメインとターゲットドメインのクラスが同じ**場合のドメイン適応。  
- データ分布は異なるが、分類すべきカテゴリが一致している。  

**例:**  
- 手書き数字（MNIST）で学習したモデルを、ストリートビューの数字（SVHN）に適応させる。  
- 異なる天候の自動運転データで学習したモデルを適応させる。

**手法:**  
- **DANN（Domain Adversarial Neural Network）**: 特徴抽出器と識別器で敵対的学習することによってドメイン適応する手法。  
Unsupervised Domain Adaptation by Backpropagation(Yaroslav Ganin+,2015) 
- **MCD**: 2つの分類器の決定境界の不一致領域を最大最小化することによってドメイン適応する手法。  
Maximum Classifier Discrepancy for Unsupervised Domain Adaptation(Kuniaki Saito+,2018)
---

## **オープンセットドメイン適応（Open-set Domain Adaptation）**
**定義:**  
- **ターゲットドメインにソースドメインにないクラスが含まれている**場合のドメイン適応。  
- モデルが「未知のクラス」を適切に識別し、適応する必要がある。  

**例:**  
- 画像分類モデルが「猫と犬」で学習しているが、ターゲットデータに「鳥」が含まれる場合。  
- 医療診断で既存の病気を学習したが、新しい病気が出てきた場合。  

**手法:**  
- **OSBP**: 損失関数の最大最小によって未知の学習とドメイン適応を同時に行う手法。  
Open set domain adaptation by backpropagation(K. Saito+,2018)
- **DTSSM**: 2つの教師生徒構造によって安定した類似度スコアで未知の学習を行う。ドメイン適応はDANNベース。  
Dual teacher–student based separation mechanism for open set domain adaptation(Yiyang Li+,2023)
- **PSDC**: これまでの手法は未知を1つのクラスで学習していたが、複数クラス用意する手法。  
PSDC: A Prototype-Based Shared-Dummy Classifier Model for Open-Set Domain Adaptation(Z. Liu+,2023)
---
