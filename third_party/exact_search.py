import os
from beir.retrieval.search.dense.util import cos_sim, dot_score
import logging
import sys
import torch
from typing import Dict, List
import json
logger = logging.getLogger(__name__)

#Parent class for any dense model
class DenseRetrievalExactSearch:
    
    def __init__(self, model, batch_size: int = 128, d_model_path = None, dataset=None, max_seq_length_d=128,corpus_chunk_size: int = 50000, pooling='cls',**kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.d_model_path= d_model_path
        self.max_seq_length_d = max_seq_length_d
        self.dataset = dataset
        self.pooling=pooling
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True #TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.results = {}
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: List[int], 
               score_function: str,
               return_sorted: bool = False, **kwargs) -> Dict[str, Dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            print("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
            
            if self.dataset != "cqadupstack":
                save_path = f'./embeddings/beir/{self.d_model_path.split("/")[-1]}-{self.max_seq_length_d}-{self.pooling}/{self.dataset}/'
            else:
                save_path = f'.embeddings/beir/{self.d_model_path.split("/")[-1]}-{self.max_seq_length_d}-{self.pooling}/{self.dataset}/{kwargs["dataset_bm_name"]}/'
                
            file_name = f'{save_path}{batch_num+1}_{len(itr)}_{self.corpus_chunk_size}_{len(corpus)}.pt'
            if os.path.exists(file_name):
                print(f"The embeddings already existed in {file_name}!")
                sub_corpus_embeddings = torch.load(file_name)
                sub_corpus_embeddings = sub_corpus_embeddings.to(query_embeddings.device)
            #Encode chunk of corpus
            else:  
                os.makedirs(save_path, exist_ok=True)  
                sub_corpus_embeddings = self.model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar, 
                    convert_to_tensor = self.convert_to_tensor
                    )
                torch.save(sub_corpus_embeddings.cpu(), file_name)
                
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            #Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=return_sorted)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]     
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx+sub_corpus_id]
                    if corpus_id != query_id:
                        self.results[query_id][corpus_id] = score

        return self.results 
