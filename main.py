from datasets import load_dataset
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

nltk.download('punkt_tab')

ptb = load_dataset('ptb-text-only/ptb_text_only', trust_remote_code=True)

train = ptb['train']
val = ptb['validation']
test = ptb['test']

def load_glove_embeddings(glove_file):
    glove_embeddings = {}

    with open(glove_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            glove_embeddings[word] = vector

    return glove_embeddings

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

def build_vocab(sentences, special_tokens=['<PAD>', '<UNK>']):
    vocab = defaultdict(lambda: len(vocab))
    for token in special_tokens:
        vocab[token]
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        for token in tokens:
            _ = vocab[token]
    return dict(vocab)

class TextDataset(Dataset):
    def __init__(self, text_data, vocab, tokenizer):
        self.text_data = text_data
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        item = self.text_data[idx]
        text = item['sentence']
        tokens = self.tokenizer(text.lower())
        token_indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return torch.tensor(token_indices, dtype=torch.long)
    
def collate_fn(batch):
    sequences = [item for item in batch]     # Extract sequences

    padded_sequences = pad_sequence(sequences, batch_first=True)    # (batch_size, max_len, embedding_dim)

    return padded_sequences

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):  # x: (batch, seq_len, d_model)
        pos = torch.arange(x.size(1), device=x.device).view(1, x.size(1))  # (1, seq_len)
        embedding = self.pos_embedding(pos)  # (1, seq_len, d_model)
        return x + embedding    

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, num_layers, dropout=0.1):
        super(TransformerLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = LearnedPositionalEncoding(d_model)

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=num_layers
        )
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):   # x: (batch, seq_len)
        # Embedding and positional encoding
        x = self.embedding(x)   # (batch, seq_len, d_model)
        x = self.pos_encoding(x)  # (batch, seq_len, d_model)

        # Fix: Create padding mask correctly and transpose it
        padding_mask = x[:, :, 0].eq(0).transpose(0, 1)  # Add transpose(0, 1)
        
        out = self.transformer_encoder(x, src_key_padding_mask=padding_mask)    # (batch, seq_len, d_model)
        out = self.fc_out(out)
        
        return out
    
def compute_perplexity(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output.view(-1, output.shape[-1]), batch.view(-1))
            total_loss += loss.item()
            num_batches += 1

    perplexity = np.exp(total_loss / num_batches)
    return perplexity


if __name__ == '__main__':

    glove_path = 'glove.twitter.27B.25d.txt'
    glove_embeddings = load_glove_embeddings(glove_path)

    vocab = build_vocab([item['sentence'] for item in train])

    dataset = TextDataset(train, vocab, word_tokenize)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    input_dim = len(glove_embeddings)
    model_dim = 100
    n_heads = 2
    num_layers = 2
    n_classes = len(glove_embeddings)

    model = TransformerLM(input_dim, model_dim, n_heads, num_layers, n_classes)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    #Train the model
    model.train()
    for epoch in range(2):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.view(-1, output.shape[-1]), batch.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

    perplexity = compute_perplexity(model, dataloader, criterion, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Perplexity: {perplexity}")