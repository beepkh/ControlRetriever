import os
from third_party.sentencebert import SentenceBERT
import math
import json
from typing import List, Dict
from pygaggle.rerank.transformer import MonoT5
from third_party.exact_search import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader
from pygaggle.rerank.base import Query, Text
import linecache
import beir
import numpy as np

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

def evaluate_model(
    model_name_or_path: str=None,
    d_model_path: str=None,
    q_model_path: str=None,
    batch_size: int=128,
    max_seq_length_d: int=350,
    max_seq_length_q: int=350,
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

    topk_for_rerank = k_values[-1]
    print(topk_for_rerank)
    # load evaluation instruction
    if use_instruction:
        instructions, ins_list = load_instruction(dataset, beir_dir)
        ins_info = {'instructions':instructions, 'ins_list':ins_list}
    else:
        ins_info = None
    print(ins_info)
    # load retrieval model
    msts = SentenceBERT(model_name_or_path, d_model_path, q_model_path,
                        max_seq_length_d, max_seq_length_q, max_seq_length_c,
                        pooling, ins_info=ins_info)
    dmodel = DRES(msts, batch_size=batch_size, d_model_path=d_model_path, 
                  max_seq_length_d=max_seq_length_d,dataset=dataset,pooling=pooling)
    retriever = EvaluateRetrieval(dmodel, k_values=k_values,score_function=score_function)
    # load beir data
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
    
    
    ori_ndcgs = []
    ndcgs = []
    # reranker = MonoT5(pretrained_model_name_or_path='./monot5-base-msmarco', token_false='▁false', token_true ='▁true')
    reranker = MonoT5(pretrained_model_name_or_path='PATH_TO_monot5-3b-msmarco-10k', token_false='▁false', token_true ='▁true')
    
    for i, data_path in enumerate(data_paths):
        if dataset=="msmarco":
            split="dev"
        else:
            split="test"
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        if dataset == "cqadupstack":
            dataset_bm_name = f"cqadupstack-{cqa_datasets[i]}"
        else:
            dataset_bm_name = dataset
        
       
        results = retriever.retrieve(corpus, queries, bm25_result = None, dataset_bm_name=dataset_bm_name)
            
        ori_ndcg, _, _, _ = EvaluateRetrieval.evaluate(
            qrels, results, retriever.k_values
        )
               
        all_rerank_result = {}
        nq = len(queries)
        cur = 0
        for qid in queries:
            query = Query(queries[qid])
            allcids = sorted(results[qid].items(),key=lambda s:s[1], reverse=True)[:topk_for_rerank]
            cids = [cc[0] for cc in allcids]
            print(f"{cur}/{nq}:", qid)
            cur+=1
            passages = [[cid, corpus[cid]['title']+' '+corpus[cid]['text']] for cid in cids]
            texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages]
            reranked = reranker.rerank(query, texts)
            reranked.sort(key=lambda x: x.score, reverse=True)
            rerank_result = {}
            for idx in range(len(texts)):
                rerank_result[str(reranked[idx].metadata["docid"])] = math.exp(reranked[idx].score) * 100
            all_rerank_result[str(qid)] = rerank_result
        
        
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, all_rerank_result, retriever.k_values
        )
        
        ori_ndcgs.append(ori_ndcg)
        ndcgs.append(ndcg)
    
    print(ori_ndcg, ndcgs)
    ori_ndcg = {k: np.mean([score[k] for score in ori_ndcgs]) for k in ori_ndcg}    
    ndcg = {k: np.mean([score[k] for score in ndcgs]) for k in ndcg}
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"results_{dataset}.json")
    result = {
        "ori_ndcg": ori_ndcg,
        "ndcg": ndcg,
    }
    print(result)
    with open(result_path, "w") as f:
        json.dump(
            result, f, indent=4,
        )
    
    return result
