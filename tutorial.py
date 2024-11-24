import torch
import torch.nn as nn
import math

"""
Input Embedding
    Embeds the input tokens using an embedding matrix and scales the embeddings by
    the square root of the model dimension.
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
"""
class LayerNormalization(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # multiplied to get normalized output
        self.bias = nn.Parameter(torch.zeros(d_model)) # added to get normalized output

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

"""
Feed Forward
    Implements the feed-forward network for the transformer model.
    Two fully connected layers with a ReLU activation function in between.
"""
class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
