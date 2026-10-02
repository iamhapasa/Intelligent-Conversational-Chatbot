# Intelligent-Conversational-Chatbot
PyTorch-based customer-support chatbot using a Transformer encoder-decoder model, custom tokenization, conversational history, top-k/top-p sampling, response-quality filtering, and Jaccard-similarity retrieval fallback.
## Features

- Transformer-based sequence-to-sequence chatbot
- PyTorch implementation
- Custom tokenization and vocabulary generation
- Conversational history support
- Top-k and top-p response sampling
- Response-quality filtering
- Jaccard-similarity retrieval fallback
- GPU support when CUDA is available
- Validation loss, perplexity, and token-accuracy tracking

## Dataset

This project uses the Bitext customer-support dataset from Hugging Face:

`bitext/Bitext-customer-support-llm-chatbot-training-dataset`

The dataset is downloaded automatically when training starts.

## Installation

``bash
pip install torch datasets

## Usage
Run the training process:
python chatbot.py

The best model checkpoint is saved as:
chatbot_transformer_seq2seq.pt

To train and chat after training, use:
from chatbot import run_pipeline

run_pipeline(train_epochs=8, max_pairs=20000, chat_after_train=True)

## Model Architecture
The chatbot uses:
- Token and positional embeddings
- Transformer encoder-decoder layers
- Adam optimizer
- Cross-entropy loss
- Learning-rate scheduling
- Temperature-based sampling
- Top-k and top-p decoding

## Project Structure
chatbot.py                   Main chatbot and training code
chatbot_colab.ipynb          Google Colab notebook
chatbot_colab_.ipynb         Alternative notebook
accuracy score.png           Model evaluation image
project flow.png             Project workflow diagram
transformer_architecture.jpg Transformer architecture diagram

## Author
Harsh Patel
