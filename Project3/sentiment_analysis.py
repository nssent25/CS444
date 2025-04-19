'''sentiment_analysis.py
Sentiment analysis classifier using word embeddings from Amazon Fashion Reviews
Nithun Selva and Saad Khan
CS444: Deep Learning
Project 3: Word Embeddings
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from amazon_reviews import make_corpus, make_word2ind_mapping, make_ind2word_mapping, find_unique_words
import network
from layers import Dense
from sklearn.model_selection import train_test_split


class MLP(network.DeepNetwork):
    '''The VGG4 network with batch normalization added to all Conv2D layers and all non-output Dense layers.'''
    def __init__(self, input_feats_shape, dense_units=(64,32), reg=0, wt_scale=1e-3, wt_init='he'):
        '''VGG4Plus network constructor

        Parameters:
        -----------
        input_feats_shape: tuple.
            The shape of input data WITHOUT the batch dimension.
        dense_units: int.
            Number of neurons in the Dense hidden layer.
        reg: float.
            The regularization strength.
        wt_scale: float.
            The scale/standard deviation of weights/biases initialized in each layer.
        wt_init: str.
            The method used to initialize the weights/biases. Options: 'normal', 'he'.
        '''
        super().__init__(input_feats_shape=input_feats_shape, reg=reg)
        self.layers = []

        prev = None
        for i in range(len(dense_units)):
            # Dense layer
            dense = Dense(
                name=f"Dense_{i}",
                units=dense_units[i],
                prev_layer_or_block=prev,
                activation='relu',
                wt_scale=wt_scale,
                wt_init=wt_init,
                do_batch_norm=False
            )
            self.layers.append(dense)
            prev = dense

        # Output layer
        self.output_layer = Dense(
            name="Output", 
            units=2, 
            prev_layer_or_block=prev, 
            activation='softmax', 
            wt_scale=wt_scale, 
            wt_init=wt_init,
            do_batch_norm=False
        )
        self.layers.append(self.output_layer)

    def __call__(self, x):
        '''Forward pass through the net with the data samples `x`.

        Parameters:
        -----------
        x: tf.constant. tf.float32s. shape=(B, Iy, Ix, n_chans).
            Data samples.

        Returns:
        --------
        tf.constant. tf.float32s. shape=(B, C).
            Activations produced by the output layer to the data.
        '''
        for layer in self.layers:
            x = layer(x)
        return x

def load_embeddings(file_path='export/embeddings.npz'):
    """Load saved word embeddings from NPZ file"""
    loaded_embeddings = np.load(file_path)
    return loaded_embeddings['embeddings']

def generate_word_sentiment_labels(N_reviews=40000, threshold=3.0, min_occurrences=5):
    """
    Generate sentiment labels for words based on their occurrence in reviews.
    
    Parameters:
    -----------
    N_reviews: int
        Number of reviews to process
    threshold: float
        Rating threshold to classify reviews as positive (> threshold) or negative (<= threshold)
    min_occurrences: int
        Minimum number of occurrences required for a word to be included

    Returns:
    --------
    X: ndarray
        Embeddings for model input
    y: ndarray
        Labels as one-hot vectors
    words: list
        List of words corresponding to each example
    """
    # Load and process reviews
    corpus, sentence_ratings, review_ids = make_corpus(N_reviews=N_reviews)
    vocab = find_unique_words(corpus)
    word2ind = make_word2ind_mapping(vocab)
    ind2word = make_ind2word_mapping(vocab)
    
    # Count occurrences in positive and negative sentences
    pos_counts = {word: 0 for word in vocab}
    neg_counts = {word: 0 for word in vocab}
    
    for sentence, rating in zip(corpus, sentence_ratings):
        is_positive = rating > threshold
        
        for word in sentence:
            if is_positive:
                pos_counts[word] += 1
            else:
                neg_counts[word] += 1
    
    # Generate labels based on counts
    word_labels = {}
    
    for word in vocab:
        pos_count = pos_counts[word]
        neg_count = neg_counts[word]
        total_count = pos_count + neg_count
        
        # Only include words that appear at least min_occurrences times
        if total_count >= min_occurrences:
            # Label as positive (1) if it appears more in positive reviews, negative (0) otherwise
            label = 1 if pos_count > neg_count else 0
            word_labels[word] = label
    
    # Make actual data for training
    embeddings = load_embeddings()
    X_raw = []
    y = []
    words = []
    
    for word, label in word_labels.items():
        if word in word2ind:  # Make sure the word is in our vocabulary
            word_idx = word2ind[word]
            X_raw.append(embeddings[word_idx])
            y.append(label)  # This is already 0 or 1
            words.append(word)
    
    X = np.array(X_raw, dtype=np.float32)  # Explicitly set float32
    y = np.array(y, dtype=np.int32)#.reshape(-1, 1)  # Reshape to (N, 1) for compatibility
    
    return X, y, words

def analyze_results(model, X, y, words, top_n=10):
    """Analyze the sentiment classification results with softmax output"""
    # Get predictions (softmax outputs: [prob_negative, prob_positive])
    predictions = model(X)
    pred_scores = predictions.numpy()  # This is shape (n_samples, 2)
    
    # Get the probability of the positive class (index 1)
    positive_probs = pred_scores[:, 1]
    
    # Flatten the true labels if needed
    true_labels = y
    if len(true_labels.shape) > 1:
        true_labels = true_labels.flatten()
    
    # Create list with words, true labels, and prediction scores
    results = list(zip(words, true_labels, positive_probs))
    
    # Sort by prediction confidence (highest first for positive predictions)
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Most positive words (by prediction)
    print("\nTop predicted POSITIVE words:")
    positive_words = []
    positive_scores = []
    
    for i, (word, true_label, pred) in enumerate(results[:top_n]):
        print(f"{i+1}. {word}: True={true_label}, Predicted={pred:.4f}")
        positive_words.append(word)
        positive_scores.append(pred)
    
    # Most negative words (by prediction)
    print("\nTop predicted NEGATIVE words:")
    negative_words = []
    negative_scores = []
    
    for i, (word, true_label, pred) in enumerate(results[-top_n:]):
        print(f"{i+1}. {word}: True={true_label}, Predicted={pred:.4f}")
        negative_words.append(word)
        negative_scores.append(pred)
    
    # Calculate classification metrics
    y_pred = (positive_probs > 0.5).astype(int)
    accuracy = np.mean(true_labels == y_pred)
    print(f"\nOverall accuracy: {accuracy:.4f}")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot top positive words
    plt.subplot(2, 1, 1)
    plt.barh(positive_words, positive_scores, color='green', alpha=0.7)
    plt.xlabel('Positive Class Probability')
    plt.title('Top Positive Words')
    plt.xlim(0, 1)
    plt.grid(axis='x')
    
    # Plot top negative words
    plt.subplot(2, 1, 2)
    plt.barh(negative_words, negative_scores, color='red', alpha=0.7)
    plt.xlabel('Positive Class Probability (lower = more negative)')
    plt.title('Top Negative Words')
    plt.xlim(0, 1)
    plt.grid(axis='x')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize training history
    if hasattr(model, 'loss_history'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_history)
        plt.title('Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        plt.show()

def run_sentiment_analysis():
    """Run the complete sentiment analysis pipeline using MLP"""
    print("Loading word embeddings...")
    embeddings = load_embeddings()
    
    print("Generating sentiment labels from reviews...")
    word_labels, word_counts = generate_word_sentiment_labels(N_reviews=40000, min_occurrences=20)
    
    # Get vocabulary and mappings
    print("Preparing data...")
    corpus, _, _ = make_corpus(N_reviews=40000)
    vocab = find_unique_words(corpus)
    word2ind = make_word2ind_mapping(vocab)
    
    # Prepare data for training
    X, y, words = prepare_sentiment_data(embeddings, word_labels, word2ind)
    print(f"Prepared {len(X)} words for sentiment analysis")
    print(f"Input shape for MLP: {X.shape}")
    
    # Count positive and negative examples
    pos_count = np.sum(y[:, 1] == 1)
    neg_count = np.sum(y[:, 0] == 1)
    print(f"Distribution: {pos_count} positive words, {neg_count} negative words")
    
    # Train classifier
    print("Training MLP for sentiment classification...")
    model, loss_history, X_test, y_test = train_sentiment_classifier(X, y, epochs=15)
    
    # Get test words
    test_indices = np.random.choice(len(words), size=len(X_test), replace=False)
    test_words = [words[i] for i in test_indices]
    
    # Analyze results
    analyze_results(model, X_test, y_test, test_words)
    
    return model, words, word_labels, word_counts

if __name__ == "__main__":
    model, words, word_labels, word_counts = run_sentiment_analysis()