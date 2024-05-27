import os
import json
from collections import defaultdict
from typing import List, Dict
import numpy as np
import beir.util
from beir.datasets.data_loader import GenericDataLoader as BEIRGenericDataLoader

from third_party.sentencebert import SentenceBERT
from beir.retrieval.evaluation import EvaluateRetrieval
from third_party.exact_search import DenseRetrievalExactSearch as DRES

import linecache

# load instructions for each dataset in beir
def load_instruction(dataset, json_path):
    ins_list = []
    instructions = {}
    data_path = os.path.join(json_path, "instructions.jsonl")
    n_ins = len(linecache.getlines(data_path))
    for i in range(n_ins):
        json_line = linecache.getline(data_path, i+1)
        print(json_line)
        ins_info = json.loads(json_line)
        instructions[ins_info.get("_id")] = ins_info.get("text")
        if ins_info['dataset'] == dataset:
            ins_list.append(ins_info['_id'])
    return instructions, ins_list

# evaluate retrievers on BEIR
def evaluate_model(
    model_name_or_path: str=None,
    d_model_path: str=None,
    q_model_path: str=None,
    batch_size: int=128,
    max_seq_length_d: int=128,
    max_seq_length_q: int=64,
    max_seq_length_c: int=350,
    beir_dir: str=None,
    dataset: str="nfcorpus",
    output_dir: str=None,
    score_function: str = "dot",
    pooling: str="mean",
    k_values: List[int] = [1,3,5,10,20,100],
    split="test",
    sep=" ", 
    use_instruction=True,
    result_dir=None,
):
    # load evaluation instruction
    if use_instruction:
        instructions, ins_list = load_instruction(dataset, beir_dir)
        ins_info = {'instructions':instructions, 'ins_list':ins_list}
        print(ins_info)
    else:
        ins_info = None
    
    # load retrieval model
    msts = SentenceBERT(model_name_or_path, d_model_path, q_model_path,
                        max_seq_length_d, max_seq_length_q, max_seq_length_c,
                        pooling, ins_info=ins_info)
    dmodel = DRES(msts, batch_size=batch_size, d_model_path=d_model_path, 
                  max_seq_length_d=max_seq_length_d,dataset=dataset,pooling=pooling)
    
    retriever = EvaluateRetrieval(dmodel, k_values=k_values,score_function=score_function)

    data_paths = []
    data_path = os.path.join(beir_dir, dataset)
    if not os.path.isdir(data_path):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = beir.util.download_and_unzip(url, beir_dir)
    cqa_datasets = ["android","english","gaming","gis","mathematica",
                    "physics","programmers","stats","tex","unix","webmasters","wordpress"
    ]
    if "cqadupstack" in data_path:
        data_paths = [
            os.path.join(data_path, sub_dataset)
            for sub_dataset in cqa_datasets
        ]
    else:
        data_paths.append(data_path)
    
    if not os.path.isdir(data_path):
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = beir.util.download_and_unzip(url, beir_dir)
    
    ndcgs = []
    _maps = []
    recalls = []
    precisions = []
    mrrs = []
    recall_caps = []
    holes = []
    
    for i, data_path in enumerate(data_paths):
        print(f"cur data_path:{data_path}")
        if dataset=="msmarco":
            split="dev"
        else:
            split="test"
        corpus, queries, qrels = BEIRGenericDataLoader(data_folder=data_path).load(split=split)
        
        if dataset == "cqadupstack":
            dataset_bm_name = f"cqadupstack-{cqa_datasets[i]}"
        else:
            dataset_bm_name = dataset

        results = retriever.retrieve(corpus, queries, dataset_bm_name=dataset_bm_name)
        
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, results, retriever.k_values
        )
        
        mrr = EvaluateRetrieval.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
        recall_cap = EvaluateRetrieval.evaluate_custom(qrels, results, retriever.k_values, metric="recall_cap")
        hole = EvaluateRetrieval.evaluate_custom(qrels, results, retriever.k_values, metric="hole")
        
        ndcgs.append(ndcg)
        _maps.append(_map)
        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)
        recall_caps.append(recall_cap)
        holes.append(hole)
        
    ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
    _map = {k: np.mean([score[k] for score in _maps]) for k in _map}
    recall = {k: np.mean([score[k] for score in recalls]) for k in recall}
    precision = {k: np.mean([score[k] for score in precisions]) for k in precision}
    mrr = {k: np.mean([score[k] for score in mrrs]) for k in mrr}   
    recall_cap = {k: np.mean([score[k] for score in recall_caps]) for k in recall_cap}   
    hole = {k: np.mean([score[k] for score in holes]) for k in hole}    
    # save the result
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"results_{dataset}.json")
    result = {
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precicion": precision,
        "mrr": mrr
    }
    print(result)
    with open(result_path, "w") as f:
        json.dump(
            result, f, indent=4,
        )
    
    return result