import os
from collections import OrderedDict
from sentence_transformers.models import Transformer, Pooling, Dense
import torch
from torch import nn
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

# architecture of ControlRetriever
class ControlTransformer(Transformer):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None, max_seq_length_c: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None, pooling_mode='mean', use_instruction=True):
        super(ControlTransformer, self).__init__(model_name_or_path, max_seq_length, model_args, cache_dir, tokenizer_args, do_lower_case, tokenizer_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self._load_model_c(model_name_or_path, config, cache_dir)
        self.pooling = Pooling(self.get_word_embedding_dimension(), pooling_mode)
        self.project_1 = zero_module(nn.Linear(config.hidden_size, config.hidden_size))
        self.project_2 = zero_module(nn.Linear(config.hidden_size, config.hidden_size))
        self.max_seq_length_c = max_seq_length_c 
        self.use_instruction = use_instruction
        self.t_embed = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    def save(self, model_path: str):
        path = os.path.join(model_path, 'model.ckpt')
        destination = OrderedDict()
        for para_name, para_value in self.state_dict().items():
            if 'auto_model' not in para_name:
                destination[para_name] = para_value
        torch.save(destination, path)
    
    def load_ckpt(self, model_path: str):
        path = os.path.join(model_path, 'model.ckpt')
        self.load_state_dict(torch.load(path), strict=False)
     
    
    def _load_model_c(self, model_name_or_path, config, cache_dir):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model_c(model_name_or_path, config, cache_dir)
        else:
            self.control_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

    def _load_t5_model_c(self, model_name_or_path, config, cache_dir):
        """Loads the encoder model from T5"""
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.control_model = T5EncoderModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
    
    
    def instruction_encode(self, t_features):
        t_trans_features = {'input_ids': t_features['input_ids'], 'attention_mask': t_features['attention_mask']}
        if 't_token_type_ids' in t_features:
            t_trans_features['token_type_ids'] = t_features['t_token_type_ids']
        
        t_embeds_all = self.t_embed(**t_trans_features, return_dict=False)
        t_embeds = t_embeds_all[0]
        
        new_t_embeds = self.project_1(t_embeds)
        t_features['token_embeddings'] = new_t_embeds
        
        return t_features
        
    
    def forward_q(self, q_features):
        q_trans_features = {'input_ids': q_features['input_ids'], 'attention_mask': q_features['attention_mask']}
        if 'token_type_ids' in q_features:
            q_trans_features['token_type_ids'] = q_features['token_type_ids']
        
        output_states = self.auto_model(**q_trans_features, return_dict=False)
        output_tokens = output_states[0]

        q_features.update({'token_embeddings': output_tokens, 'attention_mask': q_features['attention_mask']})
        
        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            q_features.update({'all_layer_embeddings': hidden_states})

        return q_features
    
    
    def combine_features(self, t_features, q_features):
        q_attention_mask = q_features['attention_mask']
        
        q_inputs_embeds = self.auto_model.get_input_embeddings()(q_features['input_ids'])
        
        if not self.use_instruction:
            return q_inputs_embeds, q_attention_mask
        
        if len(list(q_inputs_embeds.shape)) == 2:
            q_inputs_embeds = q_inputs_embeds.unsqueeze(0)
            
        t_inputs_embeds = t_features['token_embeddings']
        if len(list(t_inputs_embeds.shape)) == 2:
            t_inputs_embeds = t_inputs_embeds.unsqueeze(0)
        t_inputs_embeds = t_inputs_embeds[:,0,:].unsqueeze(1)
        
        inputs_embeds = q_inputs_embeds + t_inputs_embeds
        
        return inputs_embeds, q_attention_mask
    
    # query encoding with trainable copy 
    def forward_c(self, t_features, q_features):
        inputs_embeds, attention_mask = self.combine_features(t_features, q_features)
        
        output_states = self.control_model(input_ids=None,
                                           attention_mask=attention_mask,
                                           inputs_embeds=inputs_embeds,
                                           return_dict=False)
        output_tokens = output_states[0]
        
        c_trans_features = {'token_embeddings': output_tokens, 'attention_mask': attention_mask}
        
        if self.control_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            c_trans_features.update({'all_layer_embeddings': hidden_states})

        return c_trans_features
        
    
    def forward(self, input_features):
        q_features, t_features = input_features['q'], input_features['t']
        with torch.no_grad():
            q_features = self.forward_q(q_features)
            q_features = self.pooling(q_features)
        
        # instruction vector
        t_features = self.instruction_encode(t_features)
        # query encoding with trainable copy 
        c_features = self.forward_c(t_features, q_features)
        c_features = self.pooling(c_features)
        
        c_sentence_embed = c_features['sentence_embedding']
        q_sentence_embed = q_features['sentence_embedding']
        
        new_c_sentence_embed = self.project_2(c_sentence_embed)
        sentence_embed = q_sentence_embed + new_c_sentence_embed
        
        features = {'q_input_ids':q_features['input_ids'], 'q_attention_mask': q_features['attention_mask']}
        features.update({'t_input_ids':t_features['input_ids'], 't_attention_mask': t_features['attention_mask']})
        features['sentence_embedding'] = sentence_embed
        
        return features
         
        
        
       