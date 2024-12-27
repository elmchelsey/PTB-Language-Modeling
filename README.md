# 243HW3

Encoder-Only Transformer
This repository contains a PyTorch implementation of an Encoder-Only Transformer. The architecture is based on the Transformer model introduced by Vaswani et al. but focuses solely on the encoder for tasks such as sequence encoding or masked language modeling.

Features
Customizable Transformer Architecture: Specify the number of layers, model dimensions, and other hyperparameters.
Positional Encoding: Implements sinusoidal positional encoding to add sequence order information to input embeddings.
Multi-Head Attention: Efficiently handles attention mechanisms with support for masking.
Feed-Forward Networks: Fully connected layers with dropout and ReLU activation.
Residual Connections: Ensures stable gradient flow and effective training.
Integration with Penn Treebank (PTB) Dataset: Uses the PTB dataset for training and evaluation.
Flexible Training Pipeline: Includes training functions with loss computation, optimizer setup, and gradient clipping.
Installation
Ensure you have Python 3.7+ and install the required dependencies:

bash
Copy code
pip install torch datasets nltk wandb
Dataset
The model uses the Penn Treebank (PTB) dataset. It is loaded via the datasets library. Ensure you have an internet connection to download the dataset.

Code Overview
Key Components
build_vocab: Generates a vocabulary from a list of sentences, assigning unique indices to each token.
InputEmbedding: Maps tokens to dense vector embeddings and scales by the model's dimensionality.
PositionalEncoding: Adds sinusoidal positional encodings to embeddings.
MultiHeadAttention: Implements scaled dot-product attention across multiple heads.
FeedForward: Applies feed-forward neural networks to encoded sequences.
ResidualConnection: Adds residual connections and layer normalization.
EncoderBlock: Combines attention, feed-forward layers, and residual connections.
Encoder: Stacks multiple EncoderBlocks.
EncoderOnlyTransformer: Combines embedding, positional encoding, encoder, and a linear projection layer.
Training Function
train_encoder_transformer: Handles model training using AdamW optimizer, cross-entropy loss, and gradient clipping.
