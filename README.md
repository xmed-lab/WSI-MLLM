# WSI-MLLM

Towards Generalizable Pathology Reports via a Multimodal LLM with the Multicenter In-Context Learning

[🤗 MMF dataset](https://huggingface.co/datasets/yili7eli/MMF) [🤗 WSI-MLLM models](https://huggingface.co/yili7eli/WSI-MLLM)

## 💡 Introduction

Pathology report generation has received increasing attention in recent years. However, existing pathology report generation methods still face two main limitations: (1) these methods utilize image-report datasets where some report contents are irrelevant to the given whole slide image (WSI), making it unreliable to train an AI model to generate such content; (2) existing methods trained on data from internal hospitals struggle to generalize to external hospitals due to significant and inevitable textual discrepancies across multiple centers. To address these challenges, we first introduce a multicenter microscopic findings (MMF) dataset, which consists of WSIs and corresponding reports of lung adenocarcinoma from multiple hospitals. The MMF dataset is designed to enable the model to generate relevant report contents and evaluate its generalization to external hospitals. Second, we propose a novel Multicenter In-Context Learning (MICL) method that effectively generalizes the model to external hospitals without the need for fine-tuning. Third, we propose a new WSI-MLLM that incorporates gigabyte-sized WSIs and their image pyramid structure into MLLMs for the first time. Experiments demonstrate that WSI-MLLM significantly outperforms existing pathology report generation methods, and MICL effectively accommodates multicenter discrepancies, achieving consistent performances across hospitals and improving BLEU-4 scores by up to 26.04%. These findings underscore the effectiveness of our WSI-MLLM and MICL in generating high-quality, generalizable pathology reports. The dataset and code will be released upon acceptance.

## Eval
python eval.py --pred records/s1/pred.jsonl --gt records/s1/gt.json
python eval.py --pred records/s2/pred.jsonl --gt records/s1/gt.json
python eval.py --pred records/s3/pred.jsonl --gt records/s1/gt.json
python eval.py --pred records/s1_en/pred.jsonl --gt records/s1_en/gt.json
python eval.py --pred records/s1_en/pred.jsonl --gt records/s1_en/gt.json
python eval.py --pred records/s1_en/pred.jsonl --gt records/s1_en/gt.json

We apply 3 different data splits to ensure stable results, where s1 indicates data split-1 (en means English).
Similarly, we released 3 models for each language in the Hugging Face.
If you want to add CONCHScore, please modify the paths of CONCH in the eval.py.

## Statements
This codebase is intended for dataset‑accompanying result evaluation only. For model training and testing, please refer to [LLaVA](https://github.com/haotian-liu/LLaVA) or [Quilt‑LLaVA](https://quilt-llava.github.io/).  For model training and testing, please refer to LLaVA or Quilt‑LLaVA — the main author has graduated and lacks access permissions or time to clean the experimental code. The main modifications are:  1. Extending single‑image input to a multi‑image pyramid structure to support WSI.  2. Retaining the class token for each tile image without other vision tokens to save memory.

## Paper and Citation
```bibtex
@article{LI2026104060,
title = {Towards generalizable pathology reports via a multimodal LLM with the multicenter in-context learning},
journal = {Medical Image Analysis},
volume = {111},
pages = {104060},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2026.104060},
url = {https://www.sciencedirect.com/science/article/pii/S1361841526001283},
author = {Yi Li and Zhihao Lin and Qixiang Zhang and Xinpeng Ding and Honglong Yang and Linjing Pi and Wei Yuan and Yongqin Wen and Linglang Guo and Qingling Zhang and Xiaomeng Li},
}
