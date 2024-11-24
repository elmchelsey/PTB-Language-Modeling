import torch
import torch.nn as nn
import math

"""
Input Embedding
    Embeds the input tokens using an embedding matrix and scales the embeddings by
    the square root of the model dimension.
"""


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
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
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len  # max sequence length (within training data)
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply the sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply the cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("pe", pe)  # register the positional encoding as a buffer

    # add the positional encoding to each token in the input sequence
    def forward(self, x):
        x = x + (self.pe[:, : x.size(1), :]).requires_grad_(False)
        return self.dropout(x)


"""
Layer Normalization
    Applies layer normalization to the input tensor.
"""


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(
            torch.ones(d_model)
        )  # multiplied to get normalized output
        self.bias = nn.Parameter(torch.zeros(d_model))  # added to get normalized output

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


"""
Feed Forward
    Implements the feed-forward network for the transformer model.
    Two fully connected layers with a ReLU activation function in between.

    The input sentence with dimension (batch_size, seq_len, d_model) is converted
    to a tensor of size (batch_size, seq_len, d_ff), then back to (batch_size, 
    seq_len, d_model).
"""


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2
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
"""


class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, h: int, dropout: float
    ):  # h is the number of heads
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        # d_model must be divisible by h, so that the embedding space can be split into
        # equal parts for each head
        assert d_model % h == 0, "d_model is not divisible by h"

        # d_k is the dimension of each head
        self.d_k = d_model // h

        # define the linear transformations for q, k, v
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # Compute attention scores
        # (batch_size, h, d_k, seq_len) -> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Mask out the attention scores for word tokens that should not be attended to
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch_size, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Multiply the attention scores by the value matrix
        return (
            attention_scores @ value
        ), attention_scores  # (batch_size, h, seq_len, d_k), (batch_size, h, seq_len, seq_len)

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)  # Q' (batch_size, seq_len, d_model)
        key = self.w_k(k)  # K' (batch_size, seq_len, d_model)
        value = self.w_v(v)  # V' (batch_size, seq_len, d_model)

        # split into h heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )  # (batch_size, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(
            1, 2
        )  # (batch_size, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )  # (batch_size, h, seq_len, d_k)

        # Return the output of the attention mechanism and the attention scores
        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )
