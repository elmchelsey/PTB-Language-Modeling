import torch
import torch.nn as nn
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
from datasets import load_dataset
import torch.nn.functional as F
import argparse
import wandb
import csv
import os
from typing import Dict, List

try:
    nltk.download('punkt')
except:
    pass

ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)

train = ptb['train']
val = ptb['validation']
test = ptb['test']

"""
Build Vocabulary
    Builds a vocabulary from a list of sentences. Returns a dictionary mapping each
    unique word to a unique integer.
    Args:
        sentences: A list of sentences to build the vocabulary from.
        special_tokens: A list of special tokens to add to the vocabulary.
    Returns:
        A dictionary mapping each unique word to a unique integer.
"""
def build_vocab(sentences, special_tokens=['<PAD>', '<UNK>']):

    sentence_count = 0
    sentence_lengths = []

    vocab = defaultdict(lambda: len(vocab))
    for token in special_tokens:
        vocab[token]
    for sentence in sentences:
        sentence_count += 1
        tokens = word_tokenize(sentence.lower())
        sentence_lengths.append(len(tokens))
        for token in tokens:
            _ = vocab[token]

    return dict(vocab)


"""
Input Embedding
    Embeds the input tokens using an embedding matrix and scales the embeddings by
    the square root of the model dimension.
    Args:
        d_model: The dimension of the model.
        vocab_size: The size of the vocabulary.
"""
class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)  # scale the embeddings by
                                                            # the square root of the model dimension
    
"""
Positional Encoding
    Computes a matrix of positional encodings based on the maximum sequence length
    and adds it to the input embeddings.
    Args:
        d_model: The dimension of the model.
        seq_len: The maximum sequence length.
        dropout: The dropout rate.
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len  # max sequence length (within training data)
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply the cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)     # register the positional encoding as a buffer

    # add the positional encoding to each token in the input sequence
    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
        return self.dropout(x)

"""
Layer Normalization
    Applies layer normalization to the input tensor.
    Args:
        d_model: The dimension of the model.
        eps: A small constant to prevent division by zero.
"""
class LayerNormalization(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

"""
Feed Forward
    Implements the feed-forward network for the transformer model.
    Two fully connected layers with a ReLU activation function in between.

    The input sentence with dimension (batch_size, seq_len, d_model) is converted
    to a tensor of size (batch_size, seq_len, d_ff), then back to (batch_size, 
    seq_len, d_model).
    Args:
        d_model: The dimension of the model.
        d_ff: The dimension of the feed-forward network.
        dropout: The dropout rate.
"""
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)    # W1 and B1
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)    # W2 and B2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))

"""
Multi Head Attention
    Implements the multi-head attention mechanism. Split in the embedding space,
    not the sequence dimension, so that each head has access to the full sentence, but 
    a different part of the embedding for each word.

    Applies attention to each head by softmaxing over the dot product of the query
    and key, then concatenates the results.

    Args:
        d_model: The dimension of the model.
        h: The number of heads.
        dropout: The dropout rate.
"""
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, 'd_model must be divisible by h'
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # Compute attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Adjust mask shape to match attention_scores
            if mask.dim() == 3:
                # If mask is (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(1)
            
            # Ensure mask matches the attention scores size exactly
            if mask.size(-1) != attention_scores.size(-1):
                # Truncate or pad mask if necessary
                mask = mask[..., :attention_scores.size(-1), :attention_scores.size(-1)]
            
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        attention_scores = attention_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear transformations
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        
        # Reshape and apply final linear transformation
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(x)

"""
Residual Connection
    Applies residual connections to the input tensor by adding the output of a sublayer
    to the original input.

    Args:
        d_model: The dimension of the model.
        dropout: The dropout rate.
"""
class ResidualConnection(nn.Module):
    def __init__(self, d_model:int, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

"""
Encoder Block
    Implements the encoder block of the transformer model.
    Args:
        self_attention_block: The self-attention block.
        feed_forward_block: The feed-forward block.
        dropout: The dropout rate.
""" 
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, feed_forward_block:FeedForward, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        d_model = self_attention_block.d_model
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # normalize before + after applying self attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        # normalize before + after applying feed forward block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
"""
Encoder
    Implements the encoder of the transformer model.
    Args:
        layers: A list of encoder blocks.
"""
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        # Get d_model from the first layer's self_attention_block
        d_model = self.layers[0].self_attention_block.d_model
        self.norm = LayerNormalization(d_model)  # Pass d_model to LayerNormalization

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
"""
Linear Layer
    Projects the encoder output into vocabulary space.
    Args:
        d_model: The dimension of the model.
        vocab_size: The size of the vocabulary.
"""
class LinearLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return self.proj(x)

"""
Encoder-Only Transformer
    Implements a simplified transformer model using only the encoder.
    Args:
        encoder: The encoder
        src_embed: The source embedding layer
        src_pos: The source positional encoding layer
        linear: The linear layer
"""
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, encoder:Encoder, src_embed:InputEmbedding, src_pos:PositionalEncoding, linear:LinearLayer):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.linear = linear

    def forward(self, src, mask):
        # Embed and add positional encoding
        src = self.src_embed(src)
        src = self.src_pos(src)
        
        # Pass through encoder
        enc_output = self.encoder(src, mask)
        
        # Project to vocabulary space
        return self.linear(enc_output)

"""
Builds the encoder-only transformer model.
    Args:
        vocab_size: The size of the vocabulary
        d_model: The dimension of the model
        h: The number of heads
        N: The number of encoder layers
        d_ff: The dimension of the feed-forward network
        dropout: The dropout rate
"""
def build_encoder_transformer(vocab_size:int, d_model:int = 512, h:int = 8, N:int = 6, d_ff:int = 2048, dropout:float = 0.1):
    # Create embedding layers
    embedding = InputEmbedding(d_model, vocab_size)

    # Create positional encoding layers
    max_seq_length = 5000
    pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the encoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # Create the linear layer
    linear_layer = LinearLayer(d_model, vocab_size)

    # Create the transformer
    transformer = EncoderOnlyTransformer(encoder, embedding, pos_encoding, linear_layer)

    # Initialize transformer weights
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

def train_encoder_transformer(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device):
    """
    Trains the encoder-only transformer model.
    """
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            src = batch['source'].to(device)
            tgt = batch['target'].to(device)
            
            # Create masked attention mask
            mask = generate_target_mask(src, device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(src, mask)
            
            # Calculate loss
            loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                           tgt.contiguous().view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch: {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                src = batch['source'].to(device)
                tgt = batch['target'].to(device)
                mask = generate_target_mask(src, device)
                
                output = model(src, mask)
                loss = criterion(output.contiguous().view(-1, output.size(-1)),
                               tgt.contiguous().view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

def generate_target_mask(tgt_input, device):
    """
    Generate mask for target sequence.
    Args:
        tgt_input: Target input tensor of shape (batch_size, seq_len)
        device: Device to create tensor on
    Returns:
        Combined mask tensor of shape (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len = tgt_input.size()
    
    # Create padding mask (batch_size, seq_len, seq_len)
    padding_mask = (tgt_input != 0).unsqueeze(1).expand(batch_size, seq_len, seq_len)
    
    # Create causal mask (seq_len, seq_len)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    causal_mask = causal_mask.expand(batch_size, seq_len, seq_len)
    
    # Combine masks
    final_mask = (~causal_mask & padding_mask)
    
    return final_mask.to(device)

class PTBDataset(Dataset):
    def __init__(self, data, vocab, max_len=128):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.data[idx]['sentence']
        tokens = word_tokenize(sentence.lower())
        
        # Convert tokens to indices
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Add start and end tokens
        indices = [self.vocab['<START>']] + indices + [self.vocab['<END>']]
        
        # Truncate if too long
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
            
        return {
            'source': torch.tensor(indices[:-1]),  # Input sequence
            'target': torch.tensor(indices[1:])    # Target sequence (shifted by 1)
        }

def collate_fn(batch):
    # Sort by source sequence length (descending)
    batch.sort(key=lambda x: len(x['source']), reverse=True)
    
    # Separate source and target sequences
    src_sequences = [x['source'] for x in batch]
    tgt_sequences = [x['target'] for x in batch]
    
    # Pad sequences
    src_padded = pad_sequence(src_sequences, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_sequences, batch_first=True, padding_value=0)
    
    return {
        'source': src_padded,
        'target': tgt_padded
    }

def calculate_and_save_perplexities(model, test_dataloader, device, output_file):
    """
    Calculates per-token perplexity for each sentence and saves to CSV file.
    
    Args:
        model: The transformer model
        test_dataloader: DataLoader for test data
        device: Device to evaluate on (cuda/cpu)
        output_file: Path to output CSV file
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['ID', 'ppl'])
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                src = batch['source'].to(device)
                tgt = batch['target'].to(device)
                
                # Create mask for the source sequence
                mask = generate_target_mask(src, device)
                
                # Forward pass through the encoder-only model
                output = model(src, mask)
                
                # Calculate loss for each token
                loss = criterion(output.contiguous().view(-1, output.size(-1)),
                               tgt.contiguous().view(-1))
                
                # Reshape loss back to (batch_size, seq_len)
                loss = loss.view(tgt.shape)
                
                # Calculate perplexity for each sentence in the batch
                for i in range(src.size(0)):
                    # Get mask for non-padding tokens
                    non_pad_mask = (tgt[i] != 0)
                    
                    # Get loss for this sentence (only for non-padding tokens)
                    sentence_loss = loss[i][non_pad_mask]
                    
                    # Calculate perplexity using the mean loss
                    if len(sentence_loss) > 0:
                        avg_loss = sentence_loss.mean().item()
                        perplexity = math.exp(avg_loss)
                        
                        # Calculate sentence index
                        sentence_idx = batch_idx * test_dataloader.batch_size + i
                        
                        # Write to CSV file with exactly two decimal places
                        writer.writerow([sentence_idx, f"{perplexity:.2f}"])

def main():
    parser = argparse.ArgumentParser(description='Train a Transformer model on PTB dataset')
    
    # Model parames
    parser.add_argument('output_file', type=str, help='Output file path for perplexities')
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the model')
    parser.add_argument('--h', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--N', type=int, default=6, help='Number of encoder/decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='Dimension of feed forward network')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training params
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')


    
    args = parser.parse_args()

    hyperparameters = {
        "d_model":args.d_model,
        "h":args.h
        
    }
    
    
    # Set CUDA device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prep data
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    vocab = build_vocab([item['sentence'] for item in train], special_tokens=special_tokens)
    
    # Create datasets and dataloaders
    train_dataset = PTBDataset(train, vocab, max_len=args.max_len)
    val_dataset = PTBDataset(val, vocab, max_len=args.max_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                              shuffle=False, collate_fn=collate_fn)
    
    # Initialize the encoder-only model
    vocab_size = len(vocab)
    model = build_encoder_transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        h=args.h,
        N=args.N,
        d_ff=args.d_ff,
        dropout=args.dropout
    )
    
    # Train the model
    train_encoder_transformer(model, train_dataloader, val_dataloader, 
                            args.num_epochs, args.learning_rate, device)
    
    # Calculate and save perplexities
    test_dataset = PTBDataset(test, vocab, max_len=args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=1, 
                               shuffle=False, collate_fn=collate_fn)
    calculate_and_save_perplexities(model, test_dataloader, device, args.output_file)

if __name__ == '__main__':
    main()
