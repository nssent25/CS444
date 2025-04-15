'''amazon_reviews.py
Functions to create, preprocess, and analyze the Amazon Fashion Reviews dataset
Nithun Selva and Saad Khan
CS444: Deep Learning
Project 3: Word Embeddings
'''
import json
import re
import numpy as np
import tensorflow as tf


def load_reviews_and_ratings(file_path='data/Amazon_Fashion.jsonl', N_reviews=1):
    '''Loads in the Amazon Fashion Reviews dataset review text and the review star rating.

    This function is provided to you. You should not need to modify it.

    Parameters:
    -----------
    file_path. str.
        Path to the dataset .jsonl file.
    N_reviews: int.
        Number of reviews to load from the file.

    Returns:
    --------
    Python list of str. len=N_reviews.
        The Amazon Fashion reviews.
    ndarray. shape=(N_reviews,).
        Star rating 0.0-5.0 of each review (5.0 is highest, 0.0 is lowest).
    '''
    ratings = np.zeros(N_reviews)
    reviews = []
    with open(file_path, 'r') as fp:
        for i in range(N_reviews):
            curr_line = fp.readline()
            curr_review = json.loads(curr_line.strip())
            # Keys: 'rating', 'title', 'text', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote',
            # 'verified_purchase'
            # rating: Star rating 0.0-5.0

            ratings[i] = curr_review['rating']
            # review: string text review of current item
            reviews.append(curr_review['text'])
    return reviews, ratings


def tokenize_words(text):
    '''Transforms a string sentence into words. Replaces contractions with same word without the apostrophe.

    This method is pre-filled for you (shouldn't require modification).

    Parameters:
    -----------
    text: string. Sentence of text.

    Returns:
    -----------
    list of strings. Words in the sentence `text`.
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    # Replaces contractions with same word without the apostrophe.
    text = re.sub('([A-Za-z]+)[\'`’]([A-Za-z]+)', r'\1'r'\2', text)
    # Now split up the words
    return pattern.findall(text.lower())


def make_corpus(N_reviews, min_sent_size=2, max_sent_len=30, verbose=False):
    '''Make the text corpus of the Amazon Fashion Reviews dataset.

    Transforms text documents (list of strings) into a list of list of words (both Python lists).
    The format is [[<sentence>], [<sentence>], ...], where <sentence> = [<word>, <word>, ...].

    For the Amazon data, this transforms a list of reviews (each is a single string) into a list of
    sentences, where each sentence is represented as a list of string words. So the elements of the
    resulting list are the i-th sentence overall. Because we lose information about which review the
    sentence comes from, we maintain a separate list that keeps track of the review number (index) from
    which the current sentence originates.

    Parameters:
    -----------
    N_reviews: int.
        Number of reviews to load from the file.
    min_sent_size: int.
        Don't add sentences LESS THAN this number of words to the corpus (skip over them). This is important because it
        removes empty sentences (bad parsing) and those with not enough word context.
    max_sent_len: int.
        If a sentence is equal or longer than this length, we trim the sentence down so it is `max_sent_len` words long
        before adding it to the corpus. This is important because it prevents the content of a very long from dominating
        the corpus.
    verbose: bool.
        If False, turn off all debug print outs.

    Returns:
    -----------
    Python list of str.
        The corpus represented as: [[<sentence>], [<sentence>], ...], where <sentence> = [<word>, <word>, ...].
    ndarray of np.float32.
        The star rating of the review associated with each sentence in the corpus.
        Example: 1st review is 3.5 stars and has 3 sentences. The ratings returned would be: [3.5, 3.5, 3.5, ...].
    ndarray of np.int32.
        The review ID associated with each sentence in the corpus.
        Example: 1st review is 3.5 stars and has 3 sentences. The review IDs returned would be: [0, 0, 0, ...].

    TODO:
    1. Load in the requested number of reviews and ratings.
    2. Split each review into sentences based on periods.
    3. Tokenize the sentence into individual word strings (via provided tokenize_words function).
    4. Make sure only sentences get added to the corpus that are above the min length and prune sentences that are too
    long.
    '''
    pass



def find_unique_words(corpus):
    '''Define the vocabulary in the corpus (unique words). Finds and returns a list of the unique words in the corpus.

    Parameters:
    -----------
    corpus: Python list of lists.
        Sentences of strings (words in each sentence).

    Returns:
    -----------
    Python list of str.
        List of unique words in the corpus.
    '''
    pass


def make_word2ind_mapping(vocab):
    '''Create dictionary that looks up a word index (int) by its string.
    Indices for each word are in the range [0, vocab_sz-1].

    Parameters:
    -----------
    vocab: Python list of str.
        Unique words in corpus.

    Returns:
    -----------
    Python dictionary. key,value pairs: str,int
    '''
    pass


def make_ind2word_mapping(vocab):
    '''Create dictionary that uses a word int code to look up its word string
    Indices for each word are in the range [0, vocab_sz-1].

    Parameters:
    -----------
    vocab: Python list of str.
        Unique words in corpus.

    Returns:
    -----------
    Python dictionary with key,value pairs: int,str
    '''
    pass


def make_target_context_word_lists(corpus, word2ind, context_win_sz=2):
    '''Make the target word array ("classes") and context word array ("training samples").

    Parameters:
    -----------
    corpus: Python list of lists of str.
        List of sentences, each of which is a list of words (str).
    word2ind: Dictionary.
        Maps word string -> int code index. Range is [0, vocab_sz-1] inclusive.
    context_win_sz: int.
        How many words to include before/after the target word in sentences for context.

    Returns:
    --------
    tf.constant. tf.int32s. shape=(N,).
        The int-coded target words in the corpus,
    tf.constant. tf.int32s. shape=(N,).
        The int-coded context words.

    Each pair of target and context words occupy the i-th position of the target and context word lists that this
    function builds up. This means that there will likely be chains/sequences of repeating target words (when a target
    word has multiple context words in the context window, which is usually the case).

    Example:
    --------
    corpus = [['neural', 'nets', 'are', 'fun'], ...]
    word2ind: 'neural':0, 'nets':1, 'are':2, 'fun':3
    context_win_sz=1
    Then we will have:
    target_words_int =  [0, 1, 1, 2, 2, 3]
    context_words_int = [1, 0, 2, 1, 3, 2]

    NOTE:
    - As the example above illustrates, the number of context words in the window is NOT constant because of sentence
    edge effects.
    - The length of target_words_int and context_words_int MUST be equal!
    '''
    pass


def get_dataset_word2vec(N_reviews=40000, verbose=False):
    '''Gets and preprocesses the Amazon Fashion Reviews dataset appropriately for training the CBOW neural network.
    This is a wrapper function to automate the functions you have already written.

    Parameters:
    -----------
    N_reviews: int.
        Number of reviews to load from the file.
    verbose: bool.
        If False, turn off all debug print outs.

    Returns:
    --------
    tf.constant. tf.int32s. shape=(N,).
        The int-coded target words in the corpus.
    tf.constant. tf.int32s. shape=(N,).
        The int-coded context words.
    Python list of str.
        The vocabulary / list of unique words in the corpus.
    '''
    pass


def get_most_similar_words(k, word_str, all_embeddings, word_str2int, eps=1e-10):
    '''Get the `k` words to the word `word_str` that have the most similar embeddings in `all_embeddings`.
    Uses the cosine similarity metric.

    Parameters:
    -----------
    k: int.
        How many words with the most similar embeddings to find?
    word_str: str.
        The query word.
    all_embeddings: ndarray. shape=(M, H).
        The embeddings extracted from the trained CBOW network and converted to NumPy ndarray.
    word_str2int: Python dictionary.
        Maps word str -> int index in the vocab.
    eps: float.
        Small number to prevent potential division by 0 in the cosine similarity metric.

    Returns:
    --------
    ndarray. shape=(k+1,)
        The indices in the vocab of the query word itself + those of the k most similar words in the vocab.
    ndarray. shape=(k+1,)
        The cosine similarity of the query word to itself + that of the k most similar words in the vocab.

    NOTE: the top_k TensorFlow function should be very helpful here.
    https://www.tensorflow.org/api_docs/python/tf/math/top_k
    -
    '''
    pass


def find_unique_word_counts(corpus, sort_by_count=True):
    '''Determine the number of unique words in the corpus along with the word counts.

    This function is provided to you. You should not need to modify it.

    Parameters:
    -----------
    corpus: Python list of lists of str.
        List of sentences, each of which is a list of words (str).
    sort_by_count: bool.
        Whether to sort the words according to their frequency.

    Returns:
    --------
    Python dictionary. str->int.
        Maps the unique words (key) to their associated count in the corpus (value).

    '''
    unique_word_counts = {}
    for sent in corpus:
        for word in sent:
            if word not in unique_word_counts:
                unique_word_counts[word] = 1
            else:
                unique_word_counts[word] += 1

    if sort_by_count:
        unique_word_counts = {key: val for key, val in sorted(unique_word_counts.items(), key=lambda item: item[1],
                                                              reverse=True)}

    return unique_word_counts
