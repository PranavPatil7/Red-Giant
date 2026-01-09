import torch
import torch.nn as nn
from typing import Tuple
import os

import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import CollatorForCLM, ParquetDataset
from model import Transformer, TransformerModelArgs
from utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype


def capture(args):

    # tunable ops
    torch.cuda.tunable.enable()
    torch.cuda.tunable.record_untuned_enable()
    torch.cuda.tunable.tuning_enable(False)

    logger.info(f"Experiment args: {args}")
    # Init
    model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

    # gen fake input
    batch_size = args.batch_size
    seq_len = args.sequence_length

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    fake_input = torch.zeros((batch_size, seq_len), dtype = torch.int64, device="cuda")

    # Set up Model
    logger.info("Setting up Model...")
    model_config = TransformerModelArgs(
            dim=4096,
            n_layers=1,
            n_heads=64,
            n_kv_heads=8,
            ffn_dim_multiplier=1.3,
            multiple_of=1024,
            rope_theta=500000,
            vocab_size=tokenizer.vocab_size,
            seq_len=args.sequence_length,
        )
    with set_default_dtype(model_dtype):
        model = Transformer(model_config)
        model = model.to("cuda")
    
    model.train()

    logger.info("Starting capture!")

    out = model(fake_input)
    out.sum().backward()

    logger.info("Capture complete!")

if __name__ == "__main__":
  tune()