import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# load spaCy's English model
nlp = spacy.load('en_core_web_sm')

# custom transformer for text preprocessing using spaCy
class SpacyPreprocessor(TransformerMixin):
    def __init__(self):
        pass

    def transform(self, X, **transform_params):
        return [self._clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def _clean_text(self, text):
        doc = nlp(text.lower())  # lowercasing and processing with spaCy
        tokens = [
            token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha
        ]  # lemmatization, remove stopwords, punctuation, and non-alphabetic characters
        return ' '.join(tokens)

# function to plot bag of words
def plot_bag_of_words(data, sentiment, title):
    vectorizer = CountVectorizer(stop_words='english', max_features=20)
    X_sentiment = data[data['sentiment'] == sentiment]['text']
    X_sentiment_transformed = vectorizer.fit_transform(X_sentiment)
    sum_words = X_sentiment_transformed.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    df_words = pd.DataFrame(words_freq, columns=['word', 'count'])
    plt.figure(figsize=(10, 5))
    sns.barplot(x='count', y='word', data=df_words)
    plt.title(title)
    plt.xlabel('Count')
    plt.ylabel('Words')
    plt.savefig(f'bag_of_words_{sentiment.lower()}.png')  # save the plot
    plt.show()  # show the plot

# load the entire dataset
data = pd.read_csv('Main-Project/all_news_sentiments.csv')

# split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['sentiment'])

# check class distribution
print(train_data['sentiment'].value_counts())

# preprocessing
X_train = train_data['text']  # text data
y_train = train_data['sentiment']  # sentiment labels
X_test = test_data['text']  # text data
y_test = test_data['sentiment']  # sentiment labels

# create a pipeline that includes text preprocessing, vectorization, and the classifier
pipeline = Pipeline([
    ('preprocess', SpacyPreprocessor()),
    ('vectorize', TfidfVectorizer()),
    ('classify', MultinomialNB())
])

# reduce the parameter grid for faster grid search
param_grid = {
    'vectorize__max_df': [0.9, 1.0],
    'vectorize__ngram_range': [(1, 1)],
    'classify__alpha': [0.1, 1.0]
}

# disable multiprocessing by setting n_jobs=1
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=1, verbose=1)
grid_search.fit(X_train, y_train)

# best parameters
print(f"best parameters: {grid_search.best_params_}")

# predict and evaluate on the test set
y_pred = grid_search.predict(X_test)
print("naive bayes accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png')  # save the plot
plt.show()  # show the plot

# plot bag of words for each sentiment
plot_bag_of_words(train_data, 'Positive', 'Bag of Words for Positive Sentiment')
plot_bag_of_words(train_data, 'Neutral', 'Bag of Words for Neutral Sentiment')
plot_bag_of_words(train_data, 'Negative', 'Bag of Words for Negative Sentiment')

# function to get sentiment using Hugging Face model
def get_sentiment(sentences, batch_size=16):
    bert_dict = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        vectors = tokenizer(batch, padding=True, max_length=65, return_tensors='pt').to(device)
        outputs = bert_model(**vectors).logits
        probs = torch.nn.functional.softmax(outputs, dim=1)
        for prob in probs:
            bert_dict.append({
                'neg': round(prob[0].item(), 3),
                'neu': round(prob[1].item(), 3),
                'pos': round(prob[2].item(), 3)
            })
    return bert_dict

# load Hugging Face model and tokenizer
MODEL_NAME = 'RashidNLP/Finance-Sentiment-Classification'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# get sentiment for test set using Hugging Face model
test_sentences = X_test.tolist()
hf_sentiments = get_sentiment(test_sentences)

# convert hf_sentiments to labels
hf_labels = []
for sentiment in hf_sentiments:
    max_sentiment = max(sentiment, key=sentiment.get)
    if max_sentiment == 'neg':
        hf_labels.append('Negative')
    elif max_sentiment == 'neu':
        hf_labels.append('Neutral')
    else:
        hf_labels.append('Positive')

# evaluate Hugging Face model
print("Hugging Face model accuracy:", accuracy_score(y_test, hf_labels))
print(classification_report(y_test, hf_labels))

# plot confusion matrix for Hugging Face model
conf_matrix_hf = confusion_matrix(y_test, hf_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_hf, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix for Hugging Face Model')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix_hf.png')  # save the plot
plt.show()  # show the plot

# Load historical prices data
historical_prices = pd.read_csv('Main-Project/all_historical_prices.csv')

# Merge historical prices with test data sentiments
test_data['publishedDate'] = pd.to_datetime(test_data['publishedDate'])
historical_prices['date'] = pd.to_datetime(historical_prices['date'])

# Merge the datasets
merged_data = pd.merge(test_data, historical_prices, left_on=['symbol', 'publishedDate'], right_on=['symbol', 'date'])

# Analyze the impact of sentiments on stock prices
def analyze_sentiment_impact(merged_data):
    sentiments = merged_data['sentiment'].unique()
    for sentiment in sentiments:
        sentiment_data = merged_data[merged_data['sentiment'] == sentiment]
        plt.figure(figsize=(14, 7))
        sns.lineplot(x='date', y='close', data=sentiment_data, label=f'{sentiment} Sentiment')
        plt.title(f'Stock Prices for {sentiment} Sentiment')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(f'stock_prices_{sentiment.lower()}.png')
        plt.show()

analyze_sentiment_impact(merged_data)
