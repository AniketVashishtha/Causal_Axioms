import random
import json
from collections import Counter
from transformers import AutoTokenizer, AutoModel, GPTNeoXConfig
from datasets import load_dataset, DatasetDict
import pickle
import re
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, GPT2RoPEConfig, GPT2NoPEModel, GPT2RoPEModel
from tqdm import tqdm
import warnings
import wandb
from transformers import AdamW
from typing import Optional
from typing import Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
)
from transformers.utils import logging
import numpy as np
import re
import torch
import nltk
import os
nltk.download('punkt')
import math
import argparse
import glob


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, n_positions, n_embd, n_layer, n_head, pos_encoding='none'):
        """
        Args:
            vocab_size (int): Size of vocabulary
            n_positions (int): Maximum sequence length
            n_embd (int): Embedding dimension
            n_layer (int): Number of transformer layers
            n_head (int): Number of attention heads
            pos_encoding (str): Type of positional encoding to use:
                - 'none': No positional encoding
                - 'learned': GPT's learned position embeddings
                - 'sinusoidal': Sinusoidal position encoding
                - 'rope': Use GPTNeoX with RoPE
        """
        super(TransformerModel, self).__init__()
        
        if pos_encoding == 'rope':
            configuration = GPT2RoPEConfig(
                n_positions=2*n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                rotary_pct=1.0,  # Use RoPE for all dimensions
                rotary_emb_base=10000,
                use_cache=False,
                # _attn_implementation="eager"
            )
        else:
            configuration = GPT2Config(
                n_positions=2 * n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                use_cache=False,
                # _attn_implementation="eager"
            )

        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}_pos={pos_encoding}"
        self.n_positions = n_positions
        self.vocab_size = vocab_size
        self.pos_encoding = pos_encoding

        self._read_in = nn.Embedding(vocab_size, n_embd)
        
        # Initialize position encoding based on selected type
        if pos_encoding == 'sinusoidal':
            self.pos_encoder = PositionalEncoding(n_embd, dropout=0.0, max_len=3500)
        else:
            self.pos_encoder = None

        # Select appropriate backbone model
        if pos_encoding == 'learned':
            self._backbone = GPT2Model(configuration)
        elif pos_encoding == 'rope':
            self._backbone = GPT2RoPEModel(configuration)
        else:
            self._backbone = GPT2NoPEModel(configuration)
            
        self._read_out = nn.Linear(n_embd, vocab_size)

    def forward(self, input_ids, attention_msk, inds=None):
        embeds = self._read_in(input_ids)
        
        # Apply positional encoding if specified
        if self.pos_encoding == 'sinusoidal':
            embeds = self.pos_encoder(embeds)
            
        output = self._backbone(inputs_embeds=embeds, attention_mask=attention_msk).last_hidden_state
        prediction = self._read_out(output)
        return prediction

def count_params(model):
    parameters = model.parameters()
    print("Parameters:", parameters)
    num_params = [np.prod(p.size()) for p in parameters]
    print(
        f"Number of parameters:{np.sum(num_params) // (10**6)}M",
    )


def evaluate(model, test_dataloader, device = "cuda", use_output_loss = True):
    """
    Evaluates `model` on test dataset

    Inputs:
        - model (BertMultiChoiceClassifierModel): Logistic Regression model to be evaluated
        - test_dataloader (torch.utils.DataLoader): A dataloader defined over the test dataset

    Returns:
        - accuracy (float): Average accuracy over the test dataset 
    """
    
    model.eval()
    model = model.to(device)
    accuracy = 0
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    ### BEGIN SOLUTION
    model = model.to(device)
    # prediction_lst = []
    # label_lst = []
    with torch.no_grad():
        for test_batch in test_dataloader:
            
            # Send all values of dicts to device
            input_ids, attn_mask, label = test_batch
            # Send all values of dicts to device
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)     
            label = label.to(device)
         
            # Step 3: Feed the input features to the model to get outputs log-probabilities
            model_outs = model(input_ids, attn_mask)

            # Step 4: Compute the loss and perform backward pass
            target = label
            output = model_outs #[:,-1]

            # target[target == 0] = -100
            
            input_length = attn_mask.sum(axis=1)
            #computing loss on final answer
            # if use_output_loss:
            last_token_pred = torch.vstack([out[length-1] for out, length in zip(model_outs, input_length)]).to(device)
            last_token_label = target
            #loss computing
            # breakpoint()
            loss = loss_fn(last_token_pred, last_token_label) #when loss is being computed on final label output
            # else:
            #     last_token_pred = torch.tensor([out[length-2].argmax(-1) for out, length in zip(model_outs, input_length)])
            #     last_token_label = torch.tensor([tgt[length-2] for tgt, length in zip(target, input_length)])
            #     loss = loss_fn(output.reshape(-1, output.shape[-1]), target.reshape(-1)) #when loss is being computed on final label output

            
            last_token_pred = torch.tensor([out[length-1].argmax(-1) for out, length in zip(model_outs, input_length)]).to(device)
            # last_token_label = torch.tensor([tgt[length-2] for tgt, length in zip(target, input_length)])
            batch_accuracy = (last_token_pred == last_token_label).float().mean()
            #appending label and prediction labels

            # numpy_pred_array = last_token_pred.numpy()
            # last_token_pred_lst = numpy_pred_array.tolist()
            # prediction_lst.extend(last_token_pred_lst)

            # numpy_label_array = last_token_label.numpy()
            # last_token_label_lst = numpy_label_array.tolist()
            # label_lst.extend(last_token_label_lst)

            # breakpoint()
            total_loss += loss.item()
            accuracy += batch_accuracy.item()

    accuracy = accuracy / len(test_dataloader)
    total_loss = total_loss/len(test_dataloader)
    ### END SOLUTION
    return accuracy, total_loss
    # , prediction_lst, label_lst
    

def train(model, train_dataloader, val_dataloader,
          lr = 1e-4, num_epochs = 3,
          device = "cuda", use_output_loss = True,
          checkpoint_freq = 5, checkpoint_dir = 'checkpoints',
          resume_from = None, save_optimizer = True):
    """
    Runs the training loop with checkpointing support.
    
    New args:
        - resume_from: Path to checkpoint to resume from
        - save_optimizer: Whether to save optimizer state in checkpoints
    """
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Initialize training state
    start_epoch = 0
    best_val_accuracy = 0
    
    # Resume from checkpoint if specified
    if resume_from is not None:
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        if save_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_accuracy = checkpoint.get('best_val_accuracy', 0)
        print(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_accuracy}")
        
        # # Log restored metrics to wandb
        # wandb.log({
        #     "epoch": start_epoch,
        #     "train_loss": checkpoint.get('train_loss', 0),
        #     "val_loss": checkpoint.get('val_loss', 0),
        #     "val_accuracy": checkpoint.get('val_accuracy', 0)
        # })

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    val_accuracy, val_loss = evaluate(model, val_dataloader, device=device)
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        model.train()
        
        for train_batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            input_ids, attn_mask, target = train_batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            target = target.to(device)
            
            model_outs = model(input_ids, attn_mask)
            
            input_length = attn_mask.sum(axis=1)
            last_token_pred = torch.vstack([out[length-1] for out, length in zip(model_outs, input_length)]).to(device)
            last_token_label = target
            
            loss = loss_fn(last_token_pred, last_token_label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # wandb.log({"loss": loss.item()})
        
        epoch_loss = epoch_loss / len(train_dataloader)
        val_accuracy, val_loss = evaluate(model, val_dataloader, device=device, use_output_loss=True)
        
        # Track best model
        is_best = val_accuracy > best_val_accuracy
        best_val_accuracy = max(val_accuracy, best_val_accuracy)

        # wandb.log({
        #     "train_epoch_loss": epoch_loss,
        #     "epoch": epoch+1,
        #     "val_accuracy": val_accuracy,
        #     "val_loss": val_loss,
        #     "best_val_accuracy": best_val_accuracy
        # })
        
        print(f"Epoch {epoch} completed | Average Training Loss: {epoch_loss} | Val Accuracy: {val_accuracy} | Val Loss: {val_loss}")
        
        # Save periodic checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'best_val_accuracy': best_val_accuracy
            }
            if save_optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
        # Always save latest checkpoint for potential resume
        latest_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss, 
            'val_accuracy': val_accuracy,
            'best_val_accuracy': best_val_accuracy
        }
        if save_optimizer:
            latest_checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(latest_checkpoint, os.path.join(checkpoint_dir, 'latest.pt'))
        
        # Save best model separately
        if is_best:
            torch.save(latest_checkpoint, os.path.join(checkpoint_dir, 'best.pt'))

    return val_accuracy


def split_sentence(sentence):
    parts = re.split(r'[.!?]', sentence)
    parts = [part.strip() for part in parts if part.strip()]
    return parts


def create_chain_sentence(parts):
    if len(parts) < 2:
        return None
    conclusion = parts[-1] + ' ?'
    statement = ' . '.join(parts[:-1]) + ' .'
    chain_sentence = f"{conclusion.strip()} [SOC] {statement.strip()} [EOC] "
    return chain_sentence


def custom_tokenizer(sentence):
    # Tokenize the sentence using word_tokenize from NLTK
    # sentence = sentence.replace
    tokens_to_not_break = [
        'causes', 'does', 'cause', 'yes', 'no', '[SOC]', '[EOC]', "d-separated", "are", "and"
    ]
    
    
    sentence = sentence.replace('.', ' .')
    sentence = sentence.replace('?', ' ?')
    sentence = sentence.replace('{', ' {')
    sentence = sentence.replace('}', ' }')
    sentence = sentence.replace(',', ' ,')
    sentence = sentence.replace(')', ' )')
    tokens = sentence.split(' ')
    custom_tokens = []
    for token in tokens:
        # breakpoint()
        # Check if the token is 'causes' or 'Does'
        if token.lower() in tokens_to_not_break:
            custom_tokens.append(token)
        else:
            # Tokenize all characters at the character level
            character_tokens = list(token)
            custom_tokens.extend(character_tokens)
    return custom_tokens

def custom_tokenizer_2(sentence):
    # Tokenize the sentence using word_tokenize from NLTK
    # sentence = sentence.replace
    tokens = nltk.word_tokenize(sentence)
    return tokens


class CausalAxiomDataset(Dataset):

  def __init__(self, documents, vocab, word2idx, max_length, labels=None):
    """
    Store dataset documents and labels here so that they can be used by
    __getitem__ to process and return the samples.

    Inputs:
      - documents (list): A list of strings containing reviews in our dataset.
      - labels (list): A list of sentiment labels (1 or 0) corresponding to each document.
      - vocab (list): A list of words present in the vocabulary
      - word2idx (dict): A dictionary that maps each word to an index.
    """

    self.documents = documents
    # self.labels = labels
    self.vocab = vocab
    self.word2idx = word2idx
    self.max_length = max_length
    self.labels = labels

  def __len__(self):
    return len(self.documents)

  def __getitem__(self, idx):
    """
    Loads and returns the features and label corresponding to the `idx` index
    in the documents and labels lists.

    Inputs:
      - idx (index): Index of the dataset example to be loaded and returned

    Returns:
      - features (numpy.ndarray): The bag of word features corresponding the document indexed by `idx`
      - label (int): The sentiment label for the `idx`th document

    Hint: You can get the document and label by doing self.documents[idx],
    self.labels[idx]. Features of the document are to be extracted via
    `get_document_bow_feature` function
    """
    # id_token_list = []
  
    #set the custom tokenizer
    sent_token = custom_tokenizer(self.documents[idx])
    # sent_token = custom_tokenizer_2(self.documents[idx])
   
    # id_token_list = [self.word2idx.get(item, item) for item in sent_token]
    id_token_list = [self.word2idx[item] for item in sent_token]
    
    id_token_list.extend([0] * (self.max_length - len(id_token_list)))
    # breakpoint()
    input_ids = torch.tensor(id_token_list)

    att_mask = []
    for element in id_token_list:
        if element != 0:
            att_mask.append(1)
        else:
            att_mask.append(0)
    att_mask = torch.tensor(att_mask)
   
    if self.labels is not None:
        label = self.labels[idx]
        label_idx = torch.tensor(self.word2idx[label])
        return input_ids, att_mask, label_idx
    else:
        return input_ids, att_mask


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the checkpoint directory.
    First checks for 'latest.pt', then looks for numbered checkpoints.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        
    Returns:
        str or None: Path to latest checkpoint, or None if no checkpoints found
    """
    # First check for latest.pt
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')
    if os.path.exists(latest_path):
        return latest_path
        
    # Look for numbered checkpoints
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
    if not checkpoint_files:
        return None
        
    # Extract epoch numbers and find latest
    epoch_numbers = [int(f.split('_')[-1].replace('.pt','')) for f in checkpoint_files]
    if not epoch_numbers:
        return None
        
    latest_epoch = max(epoch_numbers)
    return os.path.join(checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pt')

def get_args():
    parser = argparse.ArgumentParser(description='Train a transformer model for causal axioms')
    
    # Model architecture
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--embedding_size', type=int, default=512,
                        help='Size of token embeddings')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--max_positions', type=int, default=3500,
                        help='Maximum sequence length')
    parser.add_argument('--pos_encoding', type=str, default='none',
                        choices=['none', 'learned', 'sinusoidal', 'rope'],
                        help='Type of positional encoding')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda or cpu)')

    # Data paths
    parser.add_argument('--train_data', type=str, 
                        default='data/dsep_instances_formatted_175k_train_dict.pkl',
                        help='Path to training data pickle file')
    parser.add_argument('--val_data', type=str,
                        default=None,
                        help='Path to validation data pickle file')
    # Wandb config
    # parser.add_argument('--wandb_project', type=str, default='CausalAxiomsTransitivity',
    #                     help='Weights & Biases project name')
    # parser.add_argument('--wandb_entity', type=str, default='kabirahuja2431',
    #                     help='Weights & Biases entity name')

    # Checkpoint config
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                        help='Save checkpoints every N epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of training data to use for validation')

    # Modify checkpoint arguments
    parser.add_argument('--resume_from_checkpoint', type=str, default='auto',
                        help='Path to checkpoint to resume training from. Use "auto" to automatically find latest checkpoint')
    parser.add_argument('--save_optimizer', action='store_true',
                        help='Whether to save optimizer state in checkpoints')

    return parser.parse_args()

def main():
    args = get_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # wandb.init(project=args.wandb_project, entity=args.wandb_entity, 
    #            config={
    #                "NumLayers": args.num_layers,
    #                "EmbeddingSize": args.embedding_size,
    #                "AttentionHeads": args.num_heads,
    #                "LearningRate": args.learning_rate,
    #                "PosEncoding": args.pos_encoding,
    #                "Seed": args.seed
    #            })
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Automatically find latest checkpoint if resume_from_checkpoint is True
    if args.resume_from_checkpoint == 'auto':
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path:
            print(f"Found latest checkpoint: {checkpoint_path}")
            args.resume_from_checkpoint = checkpoint_path
        else:
            print("No checkpoints found, starting from scratch")
            args.resume_from_checkpoint = None
    
    # Load training data
    if ".pkl" in args.train_data:
        with open(args.train_data, 'rb') as f:
            train_data = pickle.load(f)
            train_data_sent = [f"{key} {value}" for key, value in train_data.items()]
            train_sents, train_labels = zip(*train_data.items())
            train_sents, train_labels = list(train_sents), list(train_labels)
    elif ".jsonl" in args.train_data:
        with open(args.train_data, 'r') as f:
            train_data = [json.loads(line) for line in f]
            train_data_sent = [f"{item['prompt']} {item['completion']}" for item in train_data]
            train_sents = [item['prompt'] for item in train_data]
            train_labels = [item['completion'] for item in train_data]
            train_sents, train_labels = list(train_sents), list(train_labels)
            
    else:
        raise ValueError(f"Invalid file type: {args.train_data}")
    
    # Load validation data
    if args.val_data is not None:
        print(f"Loading validation data from {args.val_data}")
        with open(args.val_data, 'rb') as f:
            val_data = pickle.load(f)
            val_sents, val_labels = zip(*val_data.items())
            val_sents, val_labels = list(val_sents), list(val_labels)

    else:
        print("No validation data provided, creating validation data from training data")
        # create validation data from training data
        val_frac = args.val_frac
        train_idxs = np.random.choice(len(train_data), size=int(len(train_data) * (1 - val_frac)), replace=False)
        val_idxs = np.setdiff1d(np.arange(len(train_data)), train_idxs)
        val_sents = [train_sents[i] for i in val_idxs]
        val_labels = [train_labels[i] for i in val_idxs]
        train_sents = [train_sents[i] for i in train_idxs]
        train_labels = [train_labels[i] for i in train_idxs]

    print("Size of Training samples:", len(train_sents))
    print("Size of Val samples:", len(val_sents))
    print(max([Counter(train_sents[idx].split())["causes"] + 1 for idx in range(len(train_sents))]))

    # Create vocabulary
    tokens = []
    for sent in tqdm(train_data_sent):
        tokens.append(custom_tokenizer(sent))

    unique_tokens = list(set(item for sublist in tokens for item in sublist))
    sorted_unique_tokens = sorted(unique_tokens)
    sorted_unique_tokens = ['[PAD]'] + sorted_unique_tokens
    token2id = {element: index for index, element in enumerate(sorted_unique_tokens)}
    id2token = {value: key for key, value in token2id.items()}
    print("VOCAB SIZE: ", len(sorted_unique_tokens))

    # Create datasets
    axiom_instance_train = CausalAxiomDataset(
        documents=train_sents,
        vocab=sorted_unique_tokens,
        word2idx=token2id,
        max_length=len(max(tokens, key=len)),
        labels=train_labels
    )

    axiom_instance_val = CausalAxiomDataset(
        documents=val_sents,
        vocab=sorted_unique_tokens,
        word2idx=token2id,
        max_length=len(max(tokens, key=len)),
        labels=val_labels
    )

    # Create dataloaders
    train_dataloader = DataLoader(axiom_instance_train, batch_size=args.batch_size)
    val_dataloader = DataLoader(axiom_instance_val, batch_size=args.batch_size)

    print('\n')
    model = TransformerModel(
        vocab_size=len(sorted_unique_tokens),
        n_positions=args.max_positions,
        n_embd=args.embedding_size,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        pos_encoding=args.pos_encoding
    )

    print("MODEL SIZE CALCULATION")
    count_params(model)
    print('\n')
    
    # Only save initial model if not resuming
    if args.resume_from_checkpoint is None:
        torch.save(model.state_dict(), f"{args.checkpoint_dir}/initial_model.pt")
    
    model.to(args.device)
    train(model, train_dataloader, val_dataloader,
          lr=args.learning_rate, 
          num_epochs=args.num_epochs,
          device=args.device, 
          use_output_loss=True,
          checkpoint_freq=args.checkpoint_freq,
          checkpoint_dir=args.checkpoint_dir,
          resume_from=args.resume_from_checkpoint,
          save_optimizer=args.save_optimizer)
    
    torch.save(model.state_dict(), f"{args.checkpoint_dir}/final_model.pt")

if __name__ == "__main__":
    main()