# DocLLM_reimplementation

This repository is the reimplemantion of [DOCLLM: A LAYOUT-AWARE GENERATIVE LANGUAGE MODEL
FOR MULTIMODAL DOCUMENT UNDERSTANDING](https://arxiv.org/pdf/2401.00908.pdf)

# Model architecture

We re-implement the model architecture based on baichuan2-7b instead (paper uses llama2-7b since they focus on English data)

The re-implemented model architecture is availabled at 
https://huggingface.co/JinghuiLuAstronaut/DocLLM_baichuan2_7b

**Note that this is an re-implementation of model architecture, all newly added parameters are random initialized, you can download the model and continue pre-training or fine-tuning.**

# Performance

We test the performance of fine-tuned DocLLM_baichuan2_7b on the in-house KIE dataset
