'''preprocess_corpus.py
Loads text data and preprocesses it into a char representation to train character-level models
Nithun Selva and Saad Khan
CS 444: Deep Learning
Project 4: Transformers
'''
import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer
import re

def load_document(path2data='data/shakespeare.txt'):
    '''Reads in the document located at `path2data` and determines the vocabulary for a character-level model.

    This function is provided to you. You should not need to modify it.

    Parameters:
    -----------
    path2data: str.
        Path to the text file that should be read in and used for the corpus.

    Returns:
    --------
    str.
        The corpus, defined as the entire document represented as a large single string.
    Python list of str.
        The vocabulary, the list of all the unique chars in the corpus.
    '''
    # Read in text file as a single string
    with open(path2data, 'r') as fp:
        corpus = fp.read()

    # Get a list of unique chars
    vocab = sorted(list(set(corpus)))

    return corpus, vocab


def make_char2ind_map(vocab):
    '''Makes the dictionary that maps char (str) → int index.

    This function is provided to you. You should not need to modify it.

    Parameters:
    -----------
    vocab: Python list of str.
        The vocabulary, the list of all the unique chars in the corpus.

    Returns:
    --------
    Dictionary mapping str → int.
        Maps each char to its position in the vocabulary.
    '''
    return dict((word, i) for i, word in enumerate(vocab))


def make_seqs_and_labels(corpus, char2ind_map, seq_len, seed=0):
    '''Makes the sequences and labels from the text corpus `corpus`, a large single string. The labels are the next
    chars for each char in the sequences.

    This function is provided to you. You should not need to modify it.

    Here is the strategy to determine the seqs and labels:
    - Walk the corpus from start to finish.
    - Process the corpus in non-overlapping segments/"windows" of `seq_len`
    - The sequences are simply the corpus chars within the current window of size `seq_len`.
    - The labels are the chars one-to-the-right of the chars grabbed for each sequence.
    - If the final few chars do not fit into a full window, just truncate and ignore them.
    - All sequences and labels should be int-coded.

    Example (without int-coding): corpus='abcdefgh'. seq_len=3.
    seq1 = ['a', 'b', 'c'], labels1 = ['b', 'c', 'd']
    seq2 = ['d', 'e', 'f'], labels2 = ['e', 'f', 'g']
    done.

    Parameters:
    -----------
    corpus: str.
        The corpus of text.
    char2ind_map: Dictionary mapping str → int.
        Maps each char to its position in the vocabulary.
    seq_len: int.
        The length of sequences of tokens to create.

    Returns:
    --------
    x_int: tf.constant. tf.int32s. shape=(num_windows, T).
        The `N` int-coded sequences with `T` time steps.
    y_int: tf.constant. tf.int32s. shape=(num_windows, T).
        The `N` int-coded labels with `T` time steps.
    '''
    # Number of nonoverlapping windows, accounting for labels being shifted shifted right by 1
    num_windows = (len(corpus) - 1) // seq_len
    x = np.zeros([num_windows, seq_len], dtype=int)
    y = np.zeros([num_windows, seq_len], dtype=int)

    for i in range(num_windows):
        x[i] = [char2ind_map[char] for char in corpus[i*seq_len:(i+1)*seq_len]]
        y[i] = [char2ind_map[char] for char in corpus[i*seq_len+1:(i+1)*seq_len+1]]

    # Shuffle
    rng = np.random.default_rng(seed=seed)  # For reproducibility
    indices = np.arange(num_windows)
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]
    print('Shuffled the sequences and labels.')

    x = tf.constant(x, dtype=tf.int32)
    y = tf.constant(y, dtype=tf.int32)
    return x, y


def make_seqs_and_labels2(corpus, char2ind_map, seq_len, stride=1, seed=0):
    '''Makes overlapping sequences and labels from the text corpus `corpus`, a large single string.
    The labels are the next chars for each char in the sequences.

    Here is the strategy to determine the seqs and labels:
    - Walk the corpus from start to finish, stepping `stride` characters at a time.
    - Each sequence is a window of `seq_len` characters.
    - The labels are the characters one-to-the-right of the characters grabbed for each sequence.
    - If the final few chars do not fit into a full window for sequences and their corresponding labels,
      they are ignored.
    - All sequences and labels should be int-coded.

    Example (without int-coding): corpus='abcdefgh'. seq_len=3.
    If stride=1:
        seq1 = ['a', 'b', 'c'], labels1 = ['b', 'c', 'd'] (start 0)
        seq2 = ['b', 'c', 'd'], labels2 = ['c', 'd', 'e'] (start 1)
        ...
        seq5 = ['e', 'f', 'g'], labels5 = ['f', 'g', 'h'] (start 4)
    If stride=2:
        seq1 = ['a', 'b', 'c'], labels1 = ['b', 'c', 'd'] (start 0)
        seq2 = ['c', 'd', 'e'], labels2 = ['d', 'e', 'f'] (start 2)
        seq3 = ['e', 'f', 'g'], labels3 = ['f', 'g', 'h'] (start 4)

    Parameters:
    -----------
    corpus: str.
        The corpus of text.
    char2ind_map: Dictionary mapping str → int.
        Maps each char to its position in the vocabulary.
    seq_len: int.
        The length of sequences of tokens to create.
    stride: int. Default is 1.
        The step size to move when creating the next sequence.
    seed: int.
        Seed for random number generator for shuffling.

    Returns:
    --------
    x_int: tf.constant. tf.int32s. shape=(num_windows, T).
        The `N` int-coded sequences with `T` time steps.
    y_int: tf.constant. tf.int32s. shape=(num_windows, T).
        The `N` int-coded labels with `T` time steps.
    '''
    # We need to be able to extract corpus[i : i + seq_len] for x
    # and corpus[i + 1 : i + seq_len + 1] for y.
    # The last character for y is at index i + seq_len.
    # So, i + seq_len must be < len(corpus).
    # i_max_start = len(corpus) - seq_len - 1.
    # If len(corpus) <= seq_len, not enough data for one sequence and its label.
    if len(corpus) <= seq_len:
        return tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32)

    # Calculate the number of windows based on stride
    # The last possible starting position `idx` for a sequence must satisfy `idx + seq_len < len(corpus)`.
    # So, `idx_max = len(corpus) - seq_len - 1`.
    # The starting positions are 0, stride, 2*stride, ..., k*stride.
    # `k*stride <= idx_max`.
    # `k <= idx_max / stride`.
    # `num_windows = floor(idx_max / stride) + 1`.
    num_windows = (len(corpus) - seq_len - 1) // stride + 1
    
    if num_windows <= 0: # Should be caught by the first check, but as a safeguard
        return tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32)

    x = np.zeros([num_windows, seq_len], dtype=int)
    y = np.zeros([num_windows, seq_len], dtype=int)

    for j in range(num_windows):
        start_index = j * stride
        x[j] = [char2ind_map[char] for char in corpus[start_index : start_index + seq_len]]
        y[j] = [char2ind_map[char] for char in corpus[start_index + 1 : start_index + seq_len + 1]]

    # Shuffle
    rng = np.random.default_rng(seed=seed)  # For reproducibility
    indices = np.arange(num_windows)
    rng.shuffle(indices)
    x = x[indices]
    y = y[indices]
    print(f'Shuffled {num_windows} overlapping sequences and labels (stride={stride}).')

    x = tf.constant(x, dtype=tf.int32)
    y = tf.constant(y, dtype=tf.int32)
    return x, y


def load_and_tokenize(path='data/shakespeare.txt', seq_len=250):
    """
    Load Shakespeare text and tokenize using GPT-2 BPE tokenizer
    
    Returns:
    --------
    x: tf.Tensor of shape (num_sequences, seq_len)
        Sequences of token IDs
    y: tf.Tensor of shape (num_sequences, seq_len)
        Target token IDs (shifted by 1)
    tokenizer: GPT2Tokenizer
        The tokenizer object for encoding/decoding tokens
    vocab_size: int
        Size of the vocabulary
    """
    # Load the corpus
    with open(path, 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    # Load the GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Add a padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_token_id = tokenizer.pad_token_id
    
    # Tokenize the entire corpus
    tokens = tokenizer.encode(corpus)
    
    # Create sequences and labels (next token prediction)
    num_sequences = (len(tokens) - 1) // seq_len
    x = np.zeros((num_sequences, seq_len), dtype=np.int32)
    y = np.zeros((num_sequences, seq_len), dtype=np.int32)
    
    for i in range(num_sequences):
        start_idx = i * seq_len
        end_idx = start_idx + seq_len
        
        # Input sequence
        x[i] = tokens[start_idx:end_idx]
        
        # Target sequence (shifted by 1)
        y[i] = tokens[start_idx+1:end_idx+1]
    
    # Convert to TensorFlow tensors
    x = tf.constant(x, dtype=tf.int32)
    y = tf.constant(y, dtype=tf.int32)
    
    return x, y, tokenizer, tokenizer.vocab_size


def normalize_lotr(raw: str) -> str:
    """Strip out backticks/extra spaces and unify quotes."""
    # 1) Remove backticks
    txt = raw.replace("`", "")
    # 2) Drop any leading spaces from each line
    txt = re.sub(r"^\s+", "", txt, flags=re.MULTILINE)
    # 3) Convert curly quotes → straight
    for curly, straight in [("‘","'"),("’","'"),("“",'"'),("”",'"')]:
        txt = txt.replace(curly, straight)
    # 4) Collapse multiple blank lines
    txt = re.sub(r"\n{2,}", "\n\n", txt)
    return txt


def load_lotr(path2data='data/LOTR.txt'):
    '''Reads in the LOTR text, normalizes it, and builds the char-vocab.'''
    with open(path2data, 'r', encoding='utf-8') as fp:
        raw = fp.read()
    # apply our cleaning pass
    corpus = normalize_lotr(raw)
    # build the sorted list of unique chars
    vocab = sorted(set(corpus))
    return corpus, vocab