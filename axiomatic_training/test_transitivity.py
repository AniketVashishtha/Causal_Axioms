import os
import argparse
import torch
import numpy as np
import random
import json
from glob import glob
from axiomatic_training.pretrain import TransformerModel
from axiomatic_training.pretrain import CausalAxiomDataset
import pickle
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
from axiomatic_training.pretrain import evaluate
from axiomatic_training.pretrain import custom_tokenizer
from axiomatic_training.pretrain import custom_tokenizer_2
from axiomatic_training.pretrain import split_sentence
from axiomatic_training.pretrain import create_chain_sentence
import sys
from tqdm import tqdm
import re

TRAIN_DATA_DIR = f"{os.getcwd()}/train_files"
EVAL_DATA_DIR = f"{os.getcwd()}/eval_transitivity"
TRAIN_DATA_PATH = f"{TRAIN_DATA_DIR}/TS2_pretrain.pkl"
BRANCHING_DATA_DIR = f"{EVAL_DATA_DIR}/branching_transitivity"
LINEAR_DATA_DIR = f"{EVAL_DATA_DIR}/simple_linear_long/simple_linear_subset_test"

MODEL_DIR = f"{os.getcwd()}/checkpoints"

def replace_o(sentence):
    # Define a regular expression pattern to match 'o' not preceded by 'D' or 'N'
    pattern = r'(?<![DN])o'
    # Replace 'o' with 'O' using the pattern
    replaced_sentence = re.sub(pattern, 'O', sentence)
    return replaced_sentence

# with open('/home/t-aniketva/Desktop/CausalAxioms/pretraining_data_final/old_pretraining_refined/updated_old_pretraining_set.pkl', 'rb') as f:
#     # Load the data from the file
#     train_data = pickle.load(f)

# with open('/home/t-aniketva/Desktop/CausalAxioms/pretraining_data_final/old_pretraining_refined/updated_old_val_set.pkl', 'rb') as f:
# # Load the data from the file
#     val_data = pickle.load(f)


def get_tokens(data_sents):
    tokens = []
    for sent in data_sents:
        tokens.append(custom_tokenizer(sent))
    return tokens

def process_data_sents(data_sents):
    processed_sents = []
    for sent in data_sents:
        # print("Before processing: ", sent)
        # tokenized = custom_tokenizer(sent) 
        # output_str = sent.replace('j', 'J')
        # output_str = output_str.replace('r', 'R')
        # output_str = output_str.replace('Q', 'q')
        output_str = sent.replace('y', 'Y')
        # output_str = output_str.replace('NO', 'N0')
        # output_str = output_str.replace('no', 'nn')
        # output_str = output_str.replace('nO', 'n0')
        # output_str = output_str.replace('m', 'a')
        # output_str = output_str.replace('6', '7')
        # output_str = output_str.replace('I', 'i')
        # # output_str = output_str.replace('o', '')
        # output_str = replace_o(output_str)
        # output_str = output_str.replace('d', 'D')
        # output_str = output_str.replace('n', 'N')
        # output_str = output_str.replace('L', 'l')
        processed_sents.append(output_str)
        # test_tokens.append(custom_tokenizer(output_str))
        #scholkopf tokenizer
    return processed_sents

def create_vocab(train_sents):
    train_tokens = get_tokens(train_sents)
    unique_tokens = list(set(item for sublist in train_tokens for item in sublist))
    sorted_unique_tokens = sorted(unique_tokens)
    sorted_unique_tokens = ['[PAD]'] + sorted_unique_tokens
    #mapping
    token2id = {element: index for index, element in enumerate(sorted_unique_tokens)}
    id2token = {value: key for key, value in token2id.items()}
    return sorted_unique_tokens, token2id, id2token

def evaluate_model(
    model,
    eval_data_sents,
    vocab,
    token2id,
    batch_size=128
):
    eval_sents, eval_labels = zip(*eval_data_sents.items())
    eval_sents = list(eval_sents)
    eval_labels = list(eval_labels)
    # eval_sents = [f"{sent} {label}" for sent, label in zip(eval_sents, eval_labels)]
    
    eval_sents = process_data_sents(eval_sents)
    eval_tokens = get_tokens(eval_sents)
    eval_axiom_dataset = CausalAxiomDataset(documents = eval_sents, 
                                            vocab = vocab, 
                                            word2idx = token2id, 
                                            max_length = len(max(eval_tokens, key=len)),
                                            labels = eval_labels)
    eval_dataloader = DataLoader(eval_axiom_dataset, batch_size=batch_size)
    accuracy, total_loss = evaluate(model, eval_dataloader, device = 'cuda')
    return accuracy, total_loss

def replace_no_nodes(data):
    
    new_data = {}
    for key, value in data.items():
        key = key.replace('no', 'n0')
        key = key.replace('NO', 'N0')
        key = key.replace('nO', 'n0')
        key = key.replace('No', 'N0')
        new_data[key] = value
    return new_data

def main(args):
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.train_data_path, 'rb') as f:
        # Load the data from the file
        train_data = pickle.load(f)        

    train_data_sent = [f"{key} {value}" for key, value in train_data.items()]
    # val_data_sent = [f"{key} {value}" for key, value in val_data.items()]
    vocab, token2id, id2token = create_vocab(train_data_sent)
    print("Token 2 id:",token2id)
    print("Vocab: ", len(vocab))
    
    model = TransformerModel(
        vocab_size = len(vocab),
        n_positions=args.n_positions, 
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        pos_encoding=args.pos_encoding
    )
    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    if args.pos_encoding == "learned" and args.ignore_learned:
        print("Ignoring learned positional encoding")
        model._backbone.wpe.weight.data = torch.zeros_like(model._backbone.wpe.weight)


    branching_accs = {}
    for file in glob(f"{BRANCHING_DATA_DIR}/*.pkl"):
        with open(file, 'rb') as f:
            branching_data = pickle.load(f)
            branching_data = replace_no_nodes(branching_data)
        branching_accuracy, branching_loss = evaluate_model(model, branching_data, vocab, token2id)
        filename = file.split("/")[-1].split(".")[0]
        print(f"Branching factor {filename} accuracy: ", branching_accuracy)
        print(f"Branching factor {filename} loss: ", branching_loss)
        print("--------------------------------")
        branching_accs[filename] = branching_accuracy
        
    simple_linear_accs = {}
    for file in glob(f"{LINEAR_DATA_DIR}/*.pkl"):
        with open(file, 'rb') as f:
            linear_data = pickle.load(f)
            linear_data = replace_no_nodes(linear_data)
        linear_accuracy, linear_loss = evaluate_model(model, linear_data, vocab, token2id)
        filename = file.split("/")[-1].split(".")[0]
        print(f"Linear factor {filename} accuracy: ", linear_accuracy)
        print(f"Linear factor {filename} loss: ", linear_loss)
        print("--------------------------------")
        simple_linear_accs[filename] = linear_accuracy
    
    
    results = {
        "test_branching": branching_accs,
        "test_linear": simple_linear_accs
    }
    
    test_file_path = f"{os.path.dirname(args.model_path)}/length_wise_test_acc.json" if not args.ignore_learned else f"{os.path.dirname(args.model_path)}/length_wise_test_acc_no_learned.json"
    with open(test_file_path, 'w') as f:
        json.dump(results, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, default=TRAIN_DATA_PATH)
    parser.add_argument("--n_positions", type=int, default=3500)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--pos_encoding", type=str, default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore_learned", action="store_true")
    args = parser.parse_args()
    main(args)