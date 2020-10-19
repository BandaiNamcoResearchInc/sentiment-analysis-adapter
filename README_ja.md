---
language: Japanese
license: MIT
---

# 感情分析用アダプター

感情分析用アダプターは、株式会社バンダイナムコ研究所がYahoo!映画のユーザレビューデータセットを用いて学習したアダプターです。
英語でのクイックスタートガイドは[こちら](README.md)をご覧ください。

## 目次

1. [イントロダクション](#イントロダクション)
1. [利用方法](#利用方法)
1. [ライセンス](#ライセンス)

## イントロダクション

巨大な事前学習済みモデルが普通となったこの時代に、ファインチューニングは当たり前な手段となっています。
しかし、ファインチューニングは一般的に良い結果が得られるものの、コストがかかります。
ファインチューニングでは、巨大な事前学習済みモデルのすべてのパラメータをエポック毎に更新する必要があり、そして更新されたすべてのパラメータを保存しなければならないため、とても非効率で再利用性がありません。
この問題は、リソースの少ないデバイスやマルチタスクの学習にとって特に重大です。

ICML2019の論文[[Parameter-Efﬁcient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)]では、アダプターモジュールを用いた転移が提案されました。
アダプターモジュールはタスク毎に学習可能なパラメータを少し追加するだけで、コンパクトで拡張性の高いモデルを実現し、以前のタスクを経ることなく新しいタスクを追加することが可能となります。
元のネットワークのパラメータは固定しているため、高い効率でパラメータを共有することができます。
GLUEタスクでは、タスクあたり3.6%のパラメータを追加するだけで、フルでファインチューニングしたモデルと同程度の性能を達成しました。

本リポジトリでは、[HuggingFace's Transformers](https://huggingface.co/transformers/index.html)や[Adapter-Hub](https://adapterhub.ml/)の素晴らしいドキュメントを基に、自身のデータセットで学習した日本語用のアダプターをどのように学習して利用するかを紹介します。


## 利用方法

### 事前学習済みのアダプターをダウンロード

[ここ](https://github.com/BandaiNamcoResearchInc/sentiment-analysis-adapter/releases)から、sentiment-analysis-adapter.zipをダウンロードし、展開します。

### アダプターのロード

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdapterType

model = AutoModelForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
model.load_adapter(adapter_path)

adapter_path = "ADAPTER_PATH" # 事前学習済みのアダプターが保存されたディレクトリのパス
```

### ポジティブな文章とネガティブな文章を分類

```python
def predict(sentence):
  token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
  input_tensor = torch.tensor([token_ids])
  outputs = model(input_tensor, adapter_names=['sst-2'])
  result = torch.argmax(outputs[0]).item()

  return 'positive' if result == 1 else 'negative'

assert predict("すばらしいですね") == "positive" 
assert predict("全然面白くない") == "negative"
```

## ライセンス
Copyright (c) 2020 BANDAI NAMCO Research Inc.

Released under the MIT license

https://opensource.org/licenses/mit-license.php