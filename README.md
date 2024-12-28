# Encoder-Only Transformer Model

This project implements an **Encoder-Only Transformer** model for natural language processing (NLP) tasks. The architecture is built using PyTorch and includes components such as multi-head attention, feed-forward layers, positional encodings, and residual connections. This project was completed in December 2024 as part of my coursework as a Natural Language Processing master's student at UCSC. 


---

## Table of Contents
1. [Dataset](#dataset)
2. [Features](#features)
3. [Model](#themodel)
4. [Hyperparameters](#hyperparameters)
5. [Acknowledgements](#acknowledgements)
---
## Dataset
This model uses the Penn Treebank (PTB) Text-Only dataset, which is loaded via the datasets library, and which contains a train, validation, and test dataset.

## Features
Vocabulary Building: Automatically constructs a vocabulary from sentences.
Input Embedding: Embeds tokens and scales by the model's dimension.
Positional Encoding: Adds positional information to embeddings.
Multi-Head Attention: Implements scaled dot-product attention across multiple heads.
Feed-Forward Layers: Two-layer MLP with ReLU activation and dropout.
Residual Connections: Includes normalization and skip connections.
Encoder Block: Combines self-attention and feed-forward layers.
Linear Projection: Maps the encoder's output to vocabulary space.

## The Model 

The build_encoder_transformer function instantiates the model. 

```python
    vocab_size = len(vocab)
    model = build_encoder_transformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        h=args.h,
        N=args.N,
        d_ff=args.d_ff,
        dropout=args.dropout
    )
```

To train the model on the PTB training dataset and evaluate its performance on the PTB test set:

```bash
python run.py
```

## Hyperparameters

The following arguments and hyperparameters can be specified via the command line:

- Output file: output_file
- Dimension of the model: --d_model
- Number of attention heads: --h
- Number of encoder layers: --N
- Dimensions of the feed-forward network: --d_ff
- Dropout: --dropout
- Batch size: --batch_size
- Number of epochs: --num_epochs
- Learning rate: --learning_rate
- Max Length: --max_len


## Acknowledgements

This project was completed using the following resources:

**Coding a Transformer from scratch on PyTorch, with full explanation, training, and inference**  
[Watch on YouTube](https://www.youtube.com/watch?v=ISNdQcPhsts&t=3685s&ab_channel=UmarJamil)  
Uploaded by **Umar Jamil**. Duration: 2:59:23.  

Zhang, Aston, et al. "Transformers."  
*Dive into Deep Learning*. 2023. [View Chapter](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html).  
Available under an [Apache 2.0 license](https://d2l.ai/license.html).  
