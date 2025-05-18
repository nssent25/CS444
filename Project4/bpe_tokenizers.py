'''
Custom BPE tokenizer implementation for text processing
'''
import re
from collections import defaultdict
import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer

def create_bpe_tokenizer(corpus, vocab_size=5000):
    """
    Create a custom BPE tokenizer for the given corpus
    
    For project purposes, we'll use the Hugging Face GPT-2 tokenizer as a base
    """
    # Initialize the GPT-2 tokenizer (pre-trained)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Add padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return tokenizer

def preprocess_with_bpe(corpus, tokenizer, seq_len=250):
    """
    Tokenize and prepare training data using BPE tokenizer
    """
    # Tokenize the corpus
    tokens = tokenizer.encode(corpus)
    
    # Create sequences and labels
    num_sequences = (len(tokens) - 1) // seq_len
    x_seqs = np.zeros((num_sequences, seq_len), dtype=np.int32)
    y_labels = np.zeros((num_sequences, seq_len), dtype=np.int32)
    
    for i in range(num_sequences):
        x_seqs[i] = tokens[i*seq_len:(i+1)*seq_len]
        y_labels[i] = tokens[i*seq_len+1:(i+1)*seq_len+1]
    
    # Convert to TensorFlow tensors
    x_seqs_tf = tf.constant(x_seqs, dtype=tf.int32)
    y_labels_tf = tf.constant(y_labels, dtype=tf.int32)
    
    return x_seqs_tf, y_labels_tf, tokenizer