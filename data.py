import re
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch

import urllib.request
import zipfile
import requests
import json
from datasets import Dataset

from torch.utils.data import DataLoader, DistributedSampler


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset


# data.py

def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=8):
    # --- LOAD DATASET ---
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    elif name == "text8":
        # FIX: Use a community hosted version of text8 that has splits
        dataset = load_dataset("afmck/text8", cache_dir=cache_dir)
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)

    # --- SELECT SPLIT ---
    if name == "lambada":
        data = dataset
    else:
        # afmck/text8 already has 'train', 'validation', 'test' splits
        # so we can access them directly via [mode] just like wikitext.
        data = dataset[mode]

    # --- SELECT DETOKENIZER ---
    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        # text8 is raw clean text (a-z space), so usually no detokenizer needed
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    EOS = tokenizer.encode(tokenizer.eos_token)[0]

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example['sentence']
        else:
            text = example["text"]
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(text, return_attention_mask=False)
        for token in tokens['input_ids']:
            token.append(EOS)
        return tokens
    
    tokenized_dataset = data.map(preprocess_and_tokenize, batched=True, num_proc=num_proc, load_from_cache_file=True)
    
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')

    # Note: Manual splitting block for text8 is removed because 
    # afmck/text8 already provides splits.
            
    return chunked_dataset


def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length)
    # Pass "validation" normally. The get_dataset function now handles the slicing for text8.
    valid_set = get_dataset(config.data.valid, "validation", cache_dir=config.data.cache_dir, block_size=config.model.length)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None
    

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
    ))
    return train_loader, valid_loader

