<h1 align = "center">
ControlRetriever: Harnessing the Power of Instructions for Controllable Retrieval
</h1>


This repo contains the PyTorch implementation of [ControlRetriever: Harnessing the Power of Instructions for Controllable Retrieval](https://arxiv.org/abs/2308.10025v1/).

## News

:star_struck: The upgrade version of ControlRetriever —— [I3: Intent-Introspective Retrieval Conditioned on Instructions](https://arxiv.org/abs/2308.10025/) has been accepted by **SIGIR 2024**.

## Installation 
This repos is built based on [beir](https://github.com/beir-cellar/beir) and [pygaggle](https://github.com/castorini/pygaggle). Please refer to the corresponding repos for the installation of the Python environment.

## Prepare
**1. Download BEIR Data**

Please first follow the [instructions](https://github.com/beir-cellar/beir/wiki/Datasets-available) to download BEIR data. Then modify the ```BEIR_DIR``` in [scripts/evaluate_retrieval.sh](scripts/evaluate_retrieval.sh#L3) and [scripts/evaluate_rerank.sh](scripts/evaluate_rerank.sh#L3) to the folder that contains the BEIR data. Finally, put [data/instructions.jsonl](data/instructions.jsonl) into the folder that contains the BEIR data.

**2. Prepare Model Checkpoints**
* Download the pretrained checkpoints of [cocodr-large](https://huggingface.co/OpenMatch/cocodr-large/tree/main). Modify the ```MODEL_NAME``` in [scripts/evaluate_retrieval.sh](scripts/evaluate_retrieval.sh#L5) and [scripts/evaluate_rerank.sh](scripts/evaluate_rerank.sh#L5) to the folder that contains cocodr-large weight.
* Download the pretrained checkpoints of [monot5-3b-msmarco-10k](https://huggingface.co/castorini/monot5-3b-msmarco-10k). Modify the corresponding context in [rerank_util.py](rerank_util.py#L91) to the folder that contains monot5-3b-msmarco-10k weight.
* Download the checkpoints of ControlRetriever from [Here](https://drive.google.com/file/d/1jKXYiAJOWqQ_7xYuwQK4-BiOF7VJQEXC/view?usp=sharing) and put ```model.ckpt``` into the ```checkpoint``` folder.

## Retrieval & Rerank
To leverage ControlRetriever for zero-shot retrieval & rerank, you can refer to the scripts provided at [scripts/evaluate_retrieval.sh](scripts/evaluate_retrieval.sh) and [scripts/evaluate_rerank.sh](scripts/evaluate_rerank.sh).

## Acknowledgment

Our project is developed based on the following repositories:

* [beir](https://github.com/beir-cellar/beir): a heterogeneous benchmark for information retrieval.
* [pygaggle](https://github.com/castorini/pygaggle): a gaggle of deep neural architectures for text ranking and question answering, designed for Pyserini.

## Citation
If you found this work useful, please consider  citing our paper as follows:
```
@inproceedings{pan2024i3,
  title={I3: I ntent-i ntrospective retrieval conditioned on i nstructions},
  author={Pan, Kaihang and Li, Juncheng and Wang, Wenjie and Fei, Hao and Song, Hongye and Ji, Wei and Lin, Jun and Liu, Xiaozhong and Chua, Tat-Seng and Tang, Siliang},
  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={1839--1849},
  year={2024}
}
```
```
@article{pan2023controlretriever,
  title={Controlretriever: Harnessing the power of instructions for controllable retrieval},
  author={Pan, Kaihang and Li, Juncheng and Song, Hongye and Fei, Hao and Ji, Wei and Zhang, Shuo and Lin, Jun and Liu, Xiaozhong and Tang, Siliang},
  journal={arXiv preprint arXiv:2308.10025},
  year={2023}
}
```
