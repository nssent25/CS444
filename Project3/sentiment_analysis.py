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
    def __init__(self, input_feats_shape, dense_units=(64,32), reg=0, wt_scale=1e-3, wt_init='he'):
        '''

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


def generate_word_sentiment_labels(N_reviews=40000, threshold=3, min_occurrences=5):
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
    word_counts = {}
    
    for word in vocab:
        pos_count = pos_counts[word]
        neg_count = neg_counts[word]
        total_count = pos_count + neg_count
        
        # Only include words that appear at least min_occurrences times
        if total_count >= min_occurrences:
            # Label as positive (1) if it appears more in positive reviews, negative (0) otherwise
            label = 1 if pos_count > neg_count else 0
            word_labels[word] = label
            word_counts[word] = total_count
    
    # Make actual data for training
    embeddings = load_embeddings()
    X_raw = []
    y = []
    words = []
    
    for word, label in word_labels.items():
        if word in word2ind:  # Make sure the word is in our vocabulary
            word_idx = word2ind[word]
            X_raw.append(embeddings[word_idx])
            y.append(label) 
            words.append(word)
    
    X = np.array(X_raw, dtype=np.float32)
    y = np.array(y, dtype=np.int32)#.reshape(-1, 1)
    
    return X, y, words, word_counts


def analyze_results(model, X, y, words, word_counts, top_n=10):
    """Analyze sentiment classification results with comprehensive metrics and visualizations"""
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Get predictions from model
    predictions = model(X)
    pred_scores = predictions.numpy()
    
    # Get class probabilities
    negative_probs = pred_scores[:, 0]
    positive_probs = pred_scores[:, 1]
    
    # Ensure y is the right shape
    true_labels = y.flatten() if len(y.shape) > 1 else y
    
    # Create binary predictions based on threshold
    y_pred = (positive_probs > 0.5).astype(int)
    
    # SECTION 1: Model Performance Metrics
    accuracy = np.mean(true_labels == y_pred)
    true_pos = np.sum((true_labels == 1) & (y_pred == 1))
    true_neg = np.sum((true_labels == 0) & (y_pred == 0))
    false_pos = np.sum((true_labels == 0) & (y_pred == 1))
    false_neg = np.sum((true_labels == 1) & (y_pred == 0))
    
    print("==== MODEL PERFORMANCE METRICS ====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"True Positives: {true_pos}")
    print(f"True Negatives: {true_neg}")
    print(f"False Positives: {false_pos}")
    print(f"False Negatives: {false_neg}")
    
    # SECTION 2: Misclassifications
    # Create result lists with all information
    results = list(zip(words, true_labels, negative_probs, positive_probs))
    
    # Find misclassified words
    false_positives = [(word, neg_prob, pos_prob) 
                       for word, true_label, neg_prob, pos_prob in results 
                       if true_label == 0 and pos_prob > 0.5]
    
    false_negatives = [(word, neg_prob, pos_prob) 
                       for word, true_label, neg_prob, pos_prob in results 
                       if true_label == 1 and pos_prob <= 0.5]
    
    # Sort by prediction confidence
    false_positives.sort(key=lambda x: x[2], reverse=True)  # highest positive prob first
    false_negatives.sort(key=lambda x: x[1], reverse=True)  # highest negative prob first
    
    print("\n==== MISCLASSIFICATIONS ====")
    print(f"Top {top_n} False Positives (negative words classified as positive):")
    for i, (word, neg_prob, pos_prob) in enumerate(false_positives[:top_n]):
        print(f"{i+1}. '{word}': Pos={pos_prob:.4f}, Count={word_counts[word]}")
    
    print(f"\nTop {top_n} False Negatives (positive words classified as negative):")
    for i, (word, neg_prob, pos_prob) in enumerate(false_negatives[:top_n]):
        print(f"{i+1}. '{word}': Neg={neg_prob:.4f}, Count={word_counts[word]}")
    
    # SECTION 3: Most Confident Words and Most Uncertain Words
    # Filter by true class
    positive_words = [(word, true_label, neg_prob, pos_prob) 
                      for word, true_label, neg_prob, pos_prob in results 
                      if true_label == 1]
    
    negative_words = [(word, true_label, neg_prob, pos_prob) 
                      for word, true_label, neg_prob, pos_prob in results 
                      if true_label == 0]
    
    # Sort for confidence levels
    most_conf_pos = sorted(positive_words, key=lambda x: x[3], reverse=True)  # highest positive prob
    most_conf_neg = sorted(negative_words, key=lambda x: x[2], reverse=True)  # highest negative prob
    
    # Find most uncertain words (closest to 50/50 prediction)
    all_words_uncertainty = [(word, true_label, neg_prob, pos_prob, abs(pos_prob - 0.5)) 
                            for word, true_label, neg_prob, pos_prob in results]
    
    # Sort by closeness to 0.5 probability
    most_uncertain = sorted(all_words_uncertainty, key=lambda x: x[4])
    
    print("\n==== WORD CONFIDENCE ANALYSIS ====")
    
    print(f"\nTop {top_n} Most Confident POSITIVE Words:")
    for i, (word, true_label, neg_prob, pos_prob) in enumerate(most_conf_pos[:top_n]):
        true_class = "Positive" if true_label == 1 else "Negative"
        print(f"{i+1}. '{word}': Pos={pos_prob:.4f}, Neg={neg_prob:.4f}, True={true_class}, Count={word_counts[word]}")
    
    print(f"\nTop {top_n} Most Confident NEGATIVE Words:")
    for i, (word, true_label, neg_prob, pos_prob) in enumerate(most_conf_neg[:top_n]):
        true_class = "Positive" if true_label == 1 else "Negative"
        print(f"{i+1}. '{word}': Neg={neg_prob:.4f}, Pos={pos_prob:.4f}, True={true_class}, Count={word_counts[word]}")
    
    print(f"\nTop {top_n} Most UNCERTAIN Words (closest to 50/50):")
    for i, (word, true_label, neg_prob, pos_prob, _) in enumerate(most_uncertain[:top_n]):
        true_class = "Positive" if true_label == 1 else "Negative"
        print(f"{i+1}. '{word}': Pos={pos_prob:.4f}, Neg={neg_prob:.4f}, True={true_class}")
    
    # SECTION 4: Visualizations
    # Extract data for plotting
    pos_words = [word for word, _, _, _ in most_conf_pos[:top_n]]
    pos_scores = [pos_prob for _, _, _, pos_prob in most_conf_pos[:top_n]]
    
    neg_words = [word for word, _, _, _ in most_conf_neg[:top_n]]
    neg_scores = [neg_prob for _, _, neg_prob, _ in most_conf_neg[:top_n]]
    
    uncertain_words = [word for word, _, _, _, _ in most_uncertain[:top_n]]
    uncertain_pos_scores = [pos_prob for _, _, _, pos_prob, _ in most_uncertain[:top_n]]
    uncertain_neg_scores = [neg_prob for _, _, neg_prob, _, _ in most_uncertain[:top_n]]
    
    
    # Confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = np.array([[true_neg, false_pos], [false_neg, true_pos]])
    plt.imshow(cm, interpolation='nearest', cmap='viridis_r')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2 + cm.min() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.tight_layout()
    plt.show()
    
    # ROC curve visualization
    fpr, tpr, _ = roc_curve(true_labels, positive_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    