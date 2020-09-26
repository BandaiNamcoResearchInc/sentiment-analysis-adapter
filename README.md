---
language: Japanese
license: MIT
---

# Sentiment Analysis Adapter
Sentiment Analysis Adapter trained on the Yahoo Movie Review dataset by Bandai Namco Research Inc.

## Table of Contents

1. [Introduction](#Introduction)
1. [Usage](#Usage)
1. [License](#License)

## Introduction

[Paper](https://arxiv.org/abs/1902.00751)

## Usage

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


## License
Copyright (c) 2020 BANDAI NAMCO Research Inc.

Released under the MIT license

https://opensource.org/licenses/mit-license.php