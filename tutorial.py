import torch
import torch.nn as nn
import math
import torch
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

nltk.download('punkt_tab')

ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)

train = ptb['train']
val = ptb['validation']
test = ptb['test']

"""
Load GloVe Embeddings
    Loads the GloVe embeddings from a file into a dictionary.
    Args:
        glove_file: The path to the GloVe embeddings file.
    Returns:
        A dictionary mapping words to their GloVe embeddings.
"""
def load_glove_embeddings(glove_file):
    glove_embeddings = {}

    with open(glove_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            glove_embeddings[word] = vector

    return glove_embeddings


"""
Preprocess Text
    Tokenizes the text, removes stopwords, and replaces unknown words with the UNK token.
    Args:
        text: The text to preprocess.
        glove_embeddings: A dictionary mapping words to their GloVe embeddings.
        unknown_token: The token to replace unknown words with.
        stopwords: A list of stopwords to remove from the text.
    Returns:
        A list of embeddings for the words in the text.
"""
def preprocess_text(text, glove_embeddings, unknown_token='UNK', stopwords=None):
    tokens = word_tokenize(text.lower())

    if stopwords is not None:
        tokens = [word for word in tokens if word not in stopwords]

    embedding_sequence = []
    for token in tokens:
        if token in glove_embeddings:
            embedding_sequence.append(glove_embeddings[token])
        else:
            embedding_sequence.append(glove_embeddings.get(unknown_token, np.zeros_like(next(iter(glove_embeddings.values())))))

    return embedding_sequence

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
    vocab = defaultdict(lambda: len(vocab))
    for token in special_tokens:
        vocab[token]
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
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

    def __init__(self, d_model:int, h:int, dropout:float):  # h is the number of heads
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        # d_model must be divisible by h, so that the embedding space can be split into 
        # equal parts for each head
        assert d_model % h == 0, 'd_model is not divisible by h'

        # d_k is the dimension of each head
        self.d_k = d_model // h

        # define the linear transformations for q, k, v
        self.w_q = nn.Linear(d_model, d_model)    # Wq
        self.w_k = nn.Linear(d_model, d_model)    # Wk
        self.w_v = nn.Linear(d_model, d_model)    # Wv

        self.w_o = nn.Linear(d_model, d_model)    # Wo
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
        attention_scores = attention_scores.softmax(dim=-1) # (batch_size, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Multiply the attention scores by the value matrix
        return (attention_scores @ value), attention_scores  # (batch_size, h, seq_len, d_k), (batch_size, h, seq_len, seq_len)

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q) # Q' (batch_size, seq_len, d_model)
        key = self.w_k(k)   # K' (batch_size, seq_len, d_model)
        value = self.w_v(v) # V' (batch_size, seq_len, d_model)

        # split into h heads
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # (batch_size, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)       # (batch_size, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)

        # Return the output of the attention mechanism and the attention scores
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Concatenate the results from all heads: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model)

        # Apply the output linear transformation
        return self.w_o(x) # (batch_size, seq_len, d_model)

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
Decoder Block
    Outlines the decoder block of the transformer
    Args:
        self_attention_block: The self-attention block.
        cross_attention_block: The cross-attention block.
        feed_forward_block: The feed-forward block.
        dropout: The dropout rate.
"""
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttention, cross_attention_block:MultiHeadAttention, feed_forward_block:FeedForward, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        d_model = self_attention_block.d_model
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, enc_output, enc_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
"""
Decoder
    Implements the decoder of the transformer model.
    Args:
        layers: A list of decoder blocks.
"""
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        # Get d_model from the first layer's self_attention_block
        d_model = self.layers[0].self_attention_block.d_model
        self.norm = LayerNormalization(d_model)  # Pass d_model to LayerNormalization

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)
    
"""
Linear
    Projects embeddings back into the vocabulary space.
    Args:
        d_model: The dimension of the model.
        vocab_size: The size of the vocabulary.
"""
class LinearLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return F.log_softmax(self.linear(x), dim=-1)
    
"""
Transformer
    Implements the entire transformer model.
    Args:
        encoder: The encoder.
        decoder: The decoder.
        src_embed: The source embedding layer.
        tgt_embed: The target embedding layer.
        src_pos: The source positional encoding layer.
        tgt_pos: The target positional encoding layer.
        linear: The linear layer.
"""
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed:InputEmbedding, tgt_embed:InputEmbedding, src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, linear:LinearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.linear = linear

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.linear(x)
    
"""
Builds the transformer model.
    Args:
        src_vocab_size: The size of the source vocabulary.
        tgt_vocab_size: The size of the target vocabulary.
        d_model: The dimension of the model.
        h: The number of heads.
        N: The number of encoder and decoder layers.
        d_ff: The dimension of the feed-forward network.
        dropout: The dropout rate.
"""
def build_transformer(src_vocab_size:int, tgt_vocab_size:int, d_model:int = 512, h:int = 8, N:int = 6, d_ff:int = 2048, dropout:float = 0.1):

    #Create embedding layers
    src_embedding = InputEmbedding(d_model, src_vocab_size)
    tgt_embedding = InputEmbedding(d_model, tgt_vocab_size)

    #Create positional encoding layers
    max_seq_length = 5000
    src_pos = PositionalEncoding(d_model, max_seq_length, dropout)
    tgt_pos = PositionalEncoding(d_model, max_seq_length, dropout)

    #Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    #Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the linear layer
    linear_layer = LinearLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, linear_layer)

    # Initialize transformer weights randomly according to the Xavier uniform distribution
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

def train_transformer(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device):
    """
    Trains the transformer model.
    Args:
        model: The transformer model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on (cuda/cpu)
    """
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is assumed to be the padding index
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Get source and target sequences
            src = batch['source'].to(device)
            tgt = batch['target'].to(device)
            
            # Create masks
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_len)
            tgt_mask = generate_square_subsequent_mask(tgt[:, :-1].size(1)).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Get model outputs
            enc_output = model.encode(src, src_mask)
            dec_output = model.decode(tgt[:, :-1], enc_output, src_mask, tgt_mask)
            output = model.project(dec_output)
            
            # Calculate loss
            loss = criterion(output.contiguous().view(-1, output.size(-1)), 
                           tgt[:, 1:].contiguous().view(-1))
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
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
                
                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
                tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
                
                enc_output = model.encode(src, src_mask)
                dec_output = model.decode(tgt[:, :-1], enc_output, src_mask, tgt_mask)
                output = model.project(dec_output)
                
                loss = criterion(output.contiguous().view(-1, output.size(-1)),
                               tgt[:, 1:].contiguous().view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f'Validation Loss: {avg_val_loss:.4f}')

def generate_square_subsequent_mask(sz):
    """
    Generates a square mask for the sequence. The mask ensures that the
    predictions for position i can depend only on the known outputs at 
    positions less than i.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

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

# Prepare the data
special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
vocab = build_vocab([item['sentence'] for item in train], special_tokens=special_tokens)

# Create datasets
train_dataset = PTBDataset(train, vocab)
val_dataset = PTBDataset(val, vocab)

# Create dataloaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize the model
vocab_size = len(vocab)
model = build_transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=512,
    h=8,
    N=6,
    d_ff=2048,
    dropout=0.1
)

# Training parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 2
learning_rate = 0.0001

# Train the model
train_transformer(model, train_dataloader, val_dataloader, num_epochs, learning_rate, device)