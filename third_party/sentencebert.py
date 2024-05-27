from sentence_transformers import SentenceTransformer
from torch import Tensor
import torch.multiprocessing as mp
from typing import List, Dict, Union, Tuple
import numpy as np
import logging
from datasets import Dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models
from third_party.MySentenceTransformer import MySentenceTransformer
from model import ControlTransformer
logger = logging.getLogger(__name__)

def concat_title_and_body(text1, text2, sep: str):
    title = text1.strip()
    body = text2.strip()
    document=[]
    if len(title):
        document.append(title)
    if len(body):
        document.append(body)
    return sep.join(document)


class SentenceBERT:
    def __init__(self, model_name_or_path, d_model_path, q_model_path=None, 
                 max_seq_length_d=350, max_seq_length_q=350, max_seq_length_c=350, 
                 pooling='mean', sep: str = " ", ins_info=None, **kwargs):
        self.sep = sep
        d_transformer = models.Transformer(
            d_model_path, max_seq_length=max_seq_length_d
        )
        d_pooling = models.Pooling(
            d_transformer.get_word_embedding_dimension(),
            pooling_mode=pooling,
        )
        self.d_model = SentenceTransformer(modules=[d_transformer, d_pooling])
        self.d_model.max_seq_length = max_seq_length_d
        print("d_model load over!")
        if ins_info == None:
            self.use_instruction = False
            q_transformer = models.Transformer(
                model_name_or_path, max_seq_length=max_seq_length_q
            )
            q_pooling = models.Pooling(
                q_transformer.get_word_embedding_dimension(),
                pooling_mode=pooling,
            )
            self.q_model = SentenceTransformer(modules=[q_transformer, q_pooling])
            self.q_model.max_seq_length = max_seq_length_q
        else:
            self.use_instruction = True
            self.instructions = ins_info['instructions']
            self.ins_list = ins_info['ins_list']

            self.query_encoder = ControlTransformer(model_name_or_path, max_seq_length=max_seq_length_q, 
                                                    max_seq_length_c=max_seq_length_c, pooling_mode=pooling, use_instruction=True)
            if q_model_path is not None:
                self.query_encoder.load_ckpt(q_model_path)
            
            self.q_model = MySentenceTransformer(modules=[self.query_encoder])


    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if self.use_instruction:
            for i in range(len(queries)):
                queries[i] = {'text':queries[i],'instruction_id':self.ins_list[0]} 
            print('encode with instruction!!!')
            return self.q_model.new_encode(queries, self.instructions, batch_size=batch_size, **kwargs)
        else:
            return self.q_model.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.d_model.encode(sentences, batch_size=batch_size, **kwargs)

    ## Encoding corpus in parallel
    def encode_corpus_parallel(self, corpus: Union[List[Dict[str, str]], Dataset], pool: Dict[str, str], batch_size: int = 8, chunk_id: int = None, **kwargs):
        if type(corpus) is dict:
            sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        
        if chunk_id is not None and chunk_id >= len(pool['processes']):
            output_queue = pool['output']
            output_queue.get()

        input_queue = pool['input']
        input_queue.put([chunk_id, batch_size, sentences])
    
    