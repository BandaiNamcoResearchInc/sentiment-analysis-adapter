---
language: Japanese
license: MIT
---

# Sentiment Analysis Adapter
A sentiment-analysis adapter trained on the Yahoo Movie Review dataset by Bandai Namco Research Inc.
Find here for a quickstart guidance in Japanese.

[日本語README](README_ja.md)

## Table of Contents

1. [Introduction](#Introduction)
1. [Usage](#Usage)
1. [License](#License)

## Introduction
In this era of large pre-trained models, Fine-tune is a perfectly normal operation. However, fine-tuning is a costly operation although it generally produces good results. Every parameter of the huge pre-trained model needs to be updated at each epoch and all the updated parameters have to be stored, which is very inefficient and not reusable at all. This problem is especially critical for low-resource device or multi-task learning.

This paper [[Parameter-Efﬁcient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)] from ICML 2019 proposedd transfer with adapter modules. Adapter modules yielded a compact and extensible model by adding only a few trainable parameters per task, and new tasks could be added without revisiting previous ones. The parameters of the original network remain fixed, yielding a high degree of parameter sharing. On GLUE task, adapter achieved equivalent performance to full fine-tuning model by adding only 3.6% parameters per task.

This repository is provide a sentiment-analysis adapter to classify POSITIVE and NEGATIVE comments in Japanese, which is trained on the Yahoo Movie Review dataset for 10 epochs and achived 89.1% accuracy.

This repository is also to show how to train and use this adapter based on the great documenation by [HuggingFace's Transformers](https://huggingface.co/transformers/index.html)
and [Adapter-Hub](https://adapterhub.ml/). Please find details in [examples](https://github.com/BandaiNamcoResearchInc/sentiment-analysis-adapter/tree/master/examples).

## Usage

### Download pretrained adapter
Dwnload and unzip sentiment-analysis-adapter.zip from [here](https://github.com/BandaiNamcoResearchInc/sentiment-analysis-adapter/releases).

### Load adapter

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdapterType

model = AutoModelForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
model.load_adapter(adapter_path)

adapter_path = "ADAPTER_PATH" # path where pretrained adapter was saved
```

### Classify positive and negative sentences

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

### Training demo on Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github//BandaiNamcoResearchInc/sentiment-analysis-adapter/blob/master/examples/adapter-train-demo.ipynb)

### Inferencing demo on Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github//BandaiNamcoResearchInc/sentiment-analysis-adapter/blob/master/examples/adapter-inference-demo.ipynb)


## License
Copyright (c) 2020 BANDAI NAMCO Research Inc.

Released under the MIT license

https://opensource.org/licenses/mit-license.php