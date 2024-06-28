import os
import re
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import emoji

# preprocessing
def preprocess_text(text):
    # remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # translate emojis to text
    text = emoji.demojize(text)
    # replace URLs with a space
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    # replace mentions and hashtags
    text = re.sub(r'@\w+|\#\w+', ' ', text)
    # remove unnecessary punctuations and lower case
    text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
    text = text.lower()
    return text

def tokenize(text):
    return word_tokenize(text)

def stem_text(text, stemmer=None):
    if stemmer:
        return [stemmer.stem(word) for word in text]
    return text

# load
def load_data(directory):
    reviews = []
    labels = []
    for label, sentiment in enumerate(["negative", "positive"]):
        folder = os.path.join(directory, sentiment)
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                reviews.append(file.read())
                labels.append(label)
    return reviews, labels

# vocabulary creation (bag of words initial)
def create_vocabulary(reviews, stemmer=None):
    vocabulary = set()
    for review in reviews:
        tokens = tokenize(preprocess_text(review))
        if stemmer:
            tokens = stem_text(tokens, stemmer)
        vocabulary.update(tokens)
    return vocabulary

# extract features using Bag of Words
def extract_features_bow(reviews, vocabulary, stemmer=None, binary=False):
    feature_vectors = []
    for review in reviews:
        tokens = tokenize(preprocess_text(review))
        if stemmer:
            tokens = stem_text(tokens, stemmer)
        if binary:
            features = {token: 1 for token in set(tokens) if token in vocabulary}
        else:
            features = Counter(token for token in tokens if token in vocabulary)
        feature_vectors.append(features)
    return feature_vectors

# Naive Bayes training with bag of Words
def train_naive_bayes_bow(features, labels, vocabulary, binary=False):
    num_docs = len(labels)
    num_pos = sum(labels)
    num_neg = num_docs - num_pos
    prior_pos = num_pos / num_docs
    prior_neg = num_neg / num_docs

    pos_word_counts = defaultdict(int)
    neg_word_counts = defaultdict(int)

    for feature, label in zip(features, labels):
        if label == 1:
            for word, count in feature.items():
                pos_word_counts[word] += count
        else:
            for word, count in feature.items():
                neg_word_counts[word] += count

    vocab_size = len(vocabulary)
    pos_total = sum(pos_word_counts.values())
    neg_total = sum(neg_word_counts.values())

    pos_likelihoods = {word: (count + 1) / (pos_total + vocab_size) for word, count in pos_word_counts.items()}
    neg_likelihoods = {word: (count + 1) / (neg_total + vocab_size) for word, count in neg_word_counts.items()}

    return prior_pos, prior_neg, pos_likelihoods, neg_likelihoods

# Naive Bayes prediction
def predict_naive_bayes(review, prior_pos, prior_neg, pos_likelihoods, neg_likelihoods, vocabulary, stemmer=None, binary=False):
    tokens = tokenize(preprocess_text(review))
    if stemmer:
        tokens = stem_text(tokens, stemmer)
    if binary:
        tokens = set(tokens)
    
    log_prob_pos = np.log(prior_pos)
    log_prob_neg = np.log(prior_neg)
    
    for token in tokens:
        if token in vocabulary:
            log_prob_pos += np.log(pos_likelihoods.get(token, 1 / (sum(pos_likelihoods.values()) + len(vocabulary))))
            log_prob_neg += np.log(neg_likelihoods.get(token, 1 / (sum(neg_likelihoods.values()) + len(vocabulary))))
    
    return 1 if log_prob_pos > log_prob_neg else 0

# eval
def evaluate_naive_bayes(test_reviews, test_labels, prior_pos, prior_neg, pos_likelihoods, neg_likelihoods, vocabulary, stemmer=None, binary=False):
    predictions = []
    for review in test_reviews:
        prediction = predict_naive_bayes(review, prior_pos, prior_neg, pos_likelihoods, neg_likelihoods, vocabulary, stemmer, binary)
        predictions.append(prediction)
    
    accuracy = np.mean(np.array(predictions) == np.array(test_labels))
    return accuracy, predictions

# confusion matrix
def calculate_confusion_matrix(predictions, labels):
    tp = fp = tn = fn = 0
    for pred, actual in zip(predictions, labels):
        if pred == 1 and actual == 1:
            tp += 1
        elif pred == 1 and actual == 0:
            fp += 1
        elif pred == 0 and actual == 1:
            fn += 1
        else:
            tn += 1
    return np.array([[tp, fp], [fn, tn]])

# performance
def calculate_performance(confusion_matrix):
    tp, fp, fn, tn = confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[1, 0], confusion_matrix[1, 1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, accuracy, f1_score

# main
def main():
    train_dir = os.path.join('tweet', 'train')
    test_dir = os.path.join('tweet', 'test')

    # load training and test data
    train_reviews, train_labels = load_data(train_dir)
    test_reviews, test_labels = load_data(test_dir)

    # create vocabulary
    stemmer = PorterStemmer()
    vocabulary = create_vocabulary(train_reviews)
    stemmed_vocabulary = create_vocabulary(train_reviews, stemmer)

    # extract and train
    for stem, vocab in zip([None, stemmer], [vocabulary, stemmed_vocabulary]):
        for binary in [False, True]:
            train_features = extract_features_bow(train_reviews, vocab, stem, binary)
            prior_pos, prior_neg, pos_likelihoods, neg_likelihoods = train_naive_bayes_bow(train_features, train_labels, vocab, binary)

            # eval
            accuracy, predictions = evaluate_naive_bayes(test_reviews, test_labels, prior_pos, prior_neg, pos_likelihoods, neg_likelihoods, vocab, stem, binary)
            print(f"Stemming: {bool(stem)}, Binary: {binary}, Accuracy: {accuracy}")

            # confusion matrix
            confusion_matrix = calculate_confusion_matrix(predictions, test_labels)
            print("Confusion Matrix:\n", confusion_matrix)

            # performance metrics
            precision, recall, accuracy, f1_score = calculate_performance(confusion_matrix)
            print(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}, F1 Score: {f1_score}")

            # save results and log file
            with open(f'results_stemming_{bool(stem)}_binary_{binary}.txt', 'w') as file:
                file.write(f"Accuracy: {accuracy}\n")
                file.write("Confusion Matrix:\n")
                file.write(f"{confusion_matrix}\n")
                file.write(f"Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}, F1 Score: {f1_score}\n")

if __name__ == "__main__":
    main()
