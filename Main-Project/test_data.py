import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from transformers import pipeline, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# load the datasets
all_news_sentiments = pd.read_csv('Main-Project/all_news_sentiments_v2.csv')
all_historical_prices = pd.read_csv('Main-Project/all_historical_prices.csv')

# convert dates to datetime and remove timezone if present
all_news_sentiments['publishedDate'] = pd.to_datetime(all_news_sentiments['publishedDate']).dt.tz_localize(None)
all_historical_prices['date'] = pd.to_datetime(all_historical_prices['date']).dt.tz_localize(None)

# load pre-trained sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# function to get sentiment from title
def get_sentiment(title):
    sentiment = sentiment_analyzer(title)[0]
    if sentiment['label'] == 'NEGATIVE':
        return 0
    elif sentiment['label'] == 'NEUTRAL':
        return 1
    else:
        return 2

# apply sentiment analysis on titles
all_news_sentiments['predicted_sentiment'] = all_news_sentiments['title'].apply(get_sentiment)

# map sentiment labels to integers
sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
all_news_sentiments['sentiment'] = all_news_sentiments['sentiment'].map(sentiment_mapping)

# check dataset size
print(f"total dataset size: {all_news_sentiments.shape[0]} rows")

# merge news sentiments with historical prices
merged_data = pd.merge(all_news_sentiments, all_historical_prices, how='inner', left_on=['symbol', 'publishedDate'], right_on=['ticker', 'date'])

# visualize the effect of news sentiment on stock prices
def plot_sentiment_vs_price(data, ticker):
    ticker_data = data[data['ticker'] == ticker]
    plt.figure(figsize=(14, 7))
    plt.plot(ticker_data['date'], ticker_data['close'], label='Close Price')
    
    # add scatter plot for sentiments
    colors = {0: 'red', 1: 'blue', 2: 'green'}
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    for sentiment in colors.keys():
        sentiment_data = ticker_data[ticker_data['predicted_sentiment'] == sentiment]
        plt.scatter(sentiment_data['date'], sentiment_data['close'], color=colors[sentiment], label=f'Sentiment: {sentiment_labels[sentiment]}', alpha=0.6)
    
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(f'Sentiment vs. Price for {ticker}')
    plt.legend()
    plt.show()

# example usage for a specific ticker symbol
plot_sentiment_vs_price(merged_data, 'AAPL')

# split the data into training and test sets
train_data, test_data = train_test_split(all_news_sentiments, test_size=0.2, random_state=42, stratify=all_news_sentiments['sentiment'])

# check for overlap between training and test data
train_titles = set(train_data['title'])
test_titles = set(test_data['title'])
overlap = train_titles.intersection(test_titles)
print(f"overlap between training and test data: {len(overlap)} articles")
assert len(overlap) == 0, "overlap detected between training and test data!"

# check the distribution of sentiment labels
print("training data sentiment distribution:")
print(train_data['sentiment'].value_counts())
print("\ntest data sentiment distribution:")
print(test_data['sentiment'].value_counts())

# naive bayes model with cross-validation and smote
vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['sentiment']

# apply smote to handle class imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nb_model = MultinomialNB()

cv_results = []

for train_index, val_index in skf.split(X_train_sm, y_train_sm):
    X_cv_train, X_cv_val = X_train_sm[train_index], X_train_sm[val_index]
    y_cv_train, y_cv_val = y_train_sm[train_index], y_train_sm[val_index]
    
    nb_model.fit(X_cv_train, y_cv_train)
    y_pred = nb_model.predict(X_cv_val)
    
    cv_results.append({
        'accuracy': accuracy_score(y_cv_val, y_pred),
        'precision': precision_score(y_cv_val, y_pred, average='weighted'),
        'recall': recall_score(y_cv_val, y_pred, average='weighted'),
        'f1_score': f1_score(y_cv_val, y_pred, average='weighted')
    })

cv_df = pd.DataFrame(cv_results)
print("naive bayes model cross-validation results:")
print(cv_df.mean())

# final training and evaluation on the test set
nb_model.fit(X_train_sm, y_train_sm)
X_test = vectorizer.transform(test_data['text'])
nb_predictions = nb_model.predict(X_test)

nb_accuracy = accuracy_score(test_data['sentiment'], nb_predictions)
nb_precision = precision_score(test_data['sentiment'], nb_predictions, average='weighted')
nb_recall = recall_score(test_data['sentiment'], nb_predictions, average='weighted')
nb_f1 = f1_score(test_data['sentiment'], nb_predictions, average='weighted')

print("naive bayes model evaluation on test data:")
print(classification_report(test_data['sentiment'], nb_predictions))
print("accuracy:", nb_accuracy)
print("precision:", nb_precision)
print("recall:", nb_recall)
print("f1 score:", nb_f1)

# bert model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_data['text'].tolist(), truncation=True, padding=True, max_length=512)

class StockSentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_labels = train_data['sentiment'].tolist()
test_labels = test_data['sentiment'].tolist()

train_dataset = StockSentimentDataset(train_encodings, train_labels)
test_dataset = StockSentimentDataset(test_encodings, test_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

print("bert model evaluation:")
bert_eval_result = trainer.evaluate()

# bert predictions
bert_test_predictions = trainer.predict(test_dataset)
bert_preds = torch.argmax(torch.tensor(bert_test_predictions.predictions), dim=1)

bert_accuracy = accuracy_score(test_data['sentiment'], bert_preds)
bert_precision = precision_score(test_data['sentiment'], bert_preds, average='weighted')
bert_recall = recall_score(test_data['sentiment'], bert_preds, average='weighted')
bert_f1 = f1_score(test_data['sentiment'], bert_preds, average='weighted')

print("bert model evaluation on test data:")
print(classification_report(test_data['sentiment'], bert_preds))
print("accuracy:", bert_accuracy)
print("precision:", bert_precision)
print("recall:", bert_recall)
print("f1 score:", bert_f1)

# visualization
metrics = ['accuracy', 'precision', 'recall', 'f1 score']
nb_scores = [nb_accuracy, nb_precision, nb_recall, nb_f1]
bert_scores = [bert_accuracy, bert_precision, bert_recall, bert_f1]

x = range(len(metrics))

plt.figure(figsize=(10, 5))
plt.bar(x, nb_scores, width=0.4, label='naive bayes', align='center')
plt.bar(x, bert_scores, width=0.4, label='bert', align='edge')
plt.xlabel('metrics')
plt.ylabel('scores')
plt.title('model comparison: naive bayes vs. bert')
plt.xticks(x, metrics)
plt.legend()
plt.show()
