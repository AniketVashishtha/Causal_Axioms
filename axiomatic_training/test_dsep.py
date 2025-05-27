import os
import argparse
import torch
import json
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
import statistics
import sys
from tqdm import tqdm
import re
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

TRAIN_DATA_DIR = f"{os.getcwd()}/train_files"
EVAL_DATA_DIR = f"{os.getcwd()}/eval_dsep"
TRAIN_DATA_PATH = f"{TRAIN_DATA_DIR}/dsep_instances_formatted_175k_train_dict.pkl"
VAL_DATA_PATH = f"{EVAL_DATA_DIR}/dsep_instances_formatted_25k_val_dict.pkl"
TEST_DATA_DIR = f"{EVAL_DATA_DIR}/len_eval_dsep"
# BRANCHING_DATA_DIR = f"{DATA_DIR}/subsampled_files_branching_dsep"
BRANCHING_DATA_DIR = f"{EVAL_DATA_DIR}/dsep_eval_branching_factor14"
COMPLEX_LIN_DATA_DIR = f"{EVAL_DATA_DIR}/dsep_eval_complex_linear/final"
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
    
    if ".pkl" in args.train_data:
        with open(args.train_data, 'rb') as f:
            # Load the data from the file
            train_data = pickle.load(f)
    elif ".jsonl" in args.train_data:
        with open(args.train_data, 'r') as f:
            train_data = [json.loads(line) for line in f]
            train_data = {item['prompt']: item['completion'] for item in train_data}
    # with open(VAL_DATA_PATH, 'rb') as f:
    # # Load the data from the file
    #     val_data = pickle.load(f)
        

    train_data_sent = [f"{key} {value}" for key, value in train_data.items()]
    # val_data_sent = [f"{key} {value}" for key, value in val_data.items()]
    vocab, token2id, id2token = create_vocab(train_data_sent)
    print("Token 2 id:",token2id)
    print("Vocab: ", len(vocab))
    
    

    # sys.stdout = open(f"{DATA_DIR}/3to9_LPE_shuffling_BestModel1.txt", 'w') # type: ignore
    
    
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


    branching_accs = {}
    # branching_factors = ["1.4", "2"]
    branching_lens = [5, 8, 10, 12]
    # for branching_factor in branching_factors:
    for length in branching_lens:
        with open(f"{BRANCHING_DATA_DIR}/dsep_instances_{length}_nodes_updated.txt", 'r') as f:
            # Load the data from the file
            branching_data = f.readlines()
            # breakpoint()
        # branching_data = {ex["prompt"]: ex["completion"] for ex in branching_data}
        branching_data = {" ".join(line.split()[:-1]).replace("'", ""): line.split()[-1].replace("'", "") for line in branching_data}
        branching_data = replace_no_nodes(branching_data)
        branching_accuracy, branching_loss = evaluate_model(model, branching_data, vocab, token2id)
        print(f"Branching length {length} accuracy: ", branching_accuracy)
        print(f"Branching length {length} loss: ", branching_loss)
        print("--------------------------------")
        # if branching_factor not in branching_accs:
        #     branching_accs[branching_factor] = {}
        # branching_accs[length] = branching_accuracy
        branching_accs[length] = branching_accuracy
    
    len2acc = {}
    chain_length = list(range(7, 15))
    for length in tqdm(chain_length):
        print("Number of nodes: ", length)
        acc_list = []
        loss_list = []
        
        with open(f"{COMPLEX_LIN_DATA_DIR}/dsep_instances_{length}_len_labeled.txt", 'r') as f:
            # Load the data from the file
            # test_data = [json.loads(line) for line in f]
            test_data = f.readlines()
        # test_data = {ex["prompt"]: ex["completion"] for ex in test_data}
        test_data = {" ".join(line.split()[:-1]).replace("'", ""): line.split()[-1].replace("'", "") for line in test_data}
        test_data = replace_no_nodes(test_data)
        # test_data_sent = [f"{key} {value}" for key, value in test_data.items()]
        # print("First test data instance: ",test_data_sent[0])
        # print("First train data instance: ",train_data_sent[0])
        # print("First val data instance: ",val_data_sent[0])
        
        test_accuracy, test_loss = evaluate_model(model, test_data, vocab, token2id)
        print("Test accuracy: ", test_accuracy)
        print("Test loss: ", test_loss)
        print("--------------------------------")

        acc_list.append(test_accuracy)
        loss_list.append(test_loss)
        len2acc[length] = test_accuracy

    # val_accuracy, val_loss = evaluate_model(model, val_data, vocab, token2id)
    # print("Validation accuracy: ", val_accuracy)
    # print("Validation loss: ", val_loss)
    # print("--------------------------------")

    results = {
        # "validation": val_accuracy,
        "test_branching": branching_accs,
        "test_complex_linear": len2acc
    }
    
    with open(f"{os.path.dirname(args.model_path)}/length_wise_test_acc.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Average accuracy: ", statistics.mean(acc_list))
    print("Average loss: ", statistics.mean(loss_list))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, default=TRAIN_DATA_PATH)
    parser.add_argument("--val_data", type=str, default=VAL_DATA_PATH)
    parser.add_argument("--n_positions", type=int, default=3500)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--pos_encoding", type=str, default="none")
    args = parser.parse_args()
    main(args)