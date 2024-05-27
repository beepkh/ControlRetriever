import argparse
from retrieval_utils import evaluate_model

import torch
import random
import numpy as np

def set_random_seed(seed):
    """Set new random seed."""
    np.random.seed(seed)
    random.seed(seed)
    # set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmard = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--d_model_path", type=str, default=None)
    parser.add_argument("--q_model_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_seq_length_d", type=int, default=128)
    parser.add_argument("--max_seq_length_q", type=int, default=128)
    parser.add_argument("--max_seq_length_c", type=int, default=350)
    parser.add_argument("--beir_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="nfcorpus")
    parser.add_argument("--score_function", choices=["dot", "cos_sim"], default="dot")
    parser.add_argument("--pooling", choices=["mean", "cls", "max"], default="mean")
    parser.add_argument("--k_values", nargs="+", type=int, default=[10])
    parser.add_argument("--split", type=str, choices=["train", "test", "dev"], default="test")
    parser.add_argument("--sep", type=str, default=" ")
    parser.add_argument("--use_instruction", action="store_true", default=False)
    parser.add_argument("--result_dir", type=str, default=None)
    # parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(args)
    seed = 0
    set_random_seed(seed)
    evaluate_model(**vars(args))