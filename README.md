# DocLLM_reimplementation

This repository is the reimplemantion of [DOCLLM: A LAYOUT-AWARE GENERATIVE LANGUAGE MODEL
FOR MULTIMODAL DOCUMENT UNDERSTANDING](https://arxiv.org/pdf/2401.00908.pdf)

# Model architecture

We re-implement the model architecture based on baichuan2-7b instead (paper uses llama2-7b since they focus on English data), model size 7.5B -> 9.1B

The re-implemented model architecture is availabled at 
https://huggingface.co/JinghuiLuAstronaut/DocLLM_baichuan2_7b

**Note that this is an re-implementation of model architecture, all newly added parameters are random initialized, you can download the model and continue pre-training or fine-tuning.**

# Performance

We test the performance of fine-tuned DocLLM_baichuan2_7b on the in-house KIE dataset, demonstrating that though without pre-training, it still achieves improvement.


| Model  | F-score |
| ------------- | ------------- |
| DocLLM\_baichuan2\_7b  | 76.75  |
| baichuan2\_7b | 74.95  |

# Quick start

```python

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch

# Load tokenizer and model
device = "cuda:0"
model_path = "model_path"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True, padding_side = 'left')
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True).to(device)

input_str = "公司:byd\n产品:极氪001"
## one poly corresponding to a token id while [-1,-1,-1,-1] represents masked poly
input_poly = [
  [0.1749,0.1466,0.5317,0.5486],
  [0.1749,0.1466,0.5317,0.5486],
  [0.1749,0.1466,0.5317,0.5486],
  [0.1749,0.1466,0.5317,0.5486],
  [-1,-1,-1,-1],
  [0.6545,0.2287,0.8743,0.4666],
  [0.6545,0.2287,0.8743,0.4666],
  [0.6545,0.2287,0.8743,0.4666],
  [0.6545,0.2287,0.8743,0.4666],
  [0.6545,0.2287,0.8743,0.4666],
  [0.6545,0.2287,0.8743,0.4666],
  [0.6545,0.2287,0.8743,0.4666]
  ]

input_ids = tokenizer.encode(input_str)
input_ids = torch.as_tensor(input_ids, dtype=torch.int64)
input_coordinates = torch.as_tensor(input_poly)

output = model(
    input_ids=input_ids.unsqueeze(0).to(device), 
    input_coordinates=input_coordinates.unsqueeze(0).to(device),
    )
