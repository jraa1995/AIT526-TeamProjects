import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

# load pre-trained sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# function to get sentiment from title
def get_sentiment(title):
    result = sentiment_analyzer(title)[0]
    return 0 if result['label'] == 'NEGATIVE' else (1 if result['label'] == 'NEUTRAL' else 2)

# tokenizer function
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

def main():
    global tokenizer  # Ensure tokenizer is accessible in the tokenize_function
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # load the datasets
    all_news_sentiments = pd.read_csv('Main-Project/main_data.csv')
    all_historical_prices = pd.read_csv('Main-Project/all_historical_prices.csv')

    # convert dates to datetime and remove timezone if present
    all_news_sentiments['publishedDate'] = pd.to_datetime(all_news_sentiments['publishedDate']).dt.tz_localize(None)
    all_historical_prices['date'] = pd.to_datetime(all_historical_prices['date']).dt.tz_localize(None)

    # handle NaN values in 'text' column by filling them with an empty string
    all_news_sentiments['text'] = all_news_sentiments['text'].fillna('')

    # apply sentiment analysis on titles
    all_news_sentiments['predicted_sentiment'] = all_news_sentiments['title'].apply(get_sentiment)

    # map sentiment labels to integers
    sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    all_news_sentiments['sentiment'] = all_news_sentiments['sentiment'].map(sentiment_mapping)

    # check dataset size
    print(f"Total dataset size: {all_news_sentiments.shape[0]} rows")

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

    # split the data into training and test sets
    train_data, test_data = train_test_split(all_news_sentiments, test_size=0.2, random_state=42, stratify=all_news_sentiments['sentiment'])

    # check for overlap between training and test data (allowing duplicates based on different ticker symbols)
    train_titles = set(train_data['title'] + train_data['symbol'])
    test_titles = set(test_data['title'] + test_data['symbol'])
    overlap = train_titles.intersection(test_titles)
    print(f"Overlap between training and test data: {len(overlap)} articles")

    # check the distribution of sentiment labels
    print("Training data sentiment distribution:")
    print(train_data['sentiment'].value_counts())
    print("\nTest data sentiment distribution:")
    print(test_data['sentiment'].value_counts())

    # naive Bayes model with cross-validation and SMOTE
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(train_data['text'])
    y_train = train_data['sentiment']

    # apply SMOTE to handle class imbalance
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
    print("Naive Bayes Model Cross-Validation Results:")
    print(cv_df.mean())

    # final training and evaluation on the test set
    nb_model.fit(X_train_sm, y_train_sm)
    X_test = vectorizer.transform(test_data['text'])
    nb_predictions = nb_model.predict(X_test)

    nb_accuracy = accuracy_score(test_data['sentiment'], nb_predictions)
    nb_precision = precision_score(test_data['sentiment'], nb_predictions, average='weighted')
    nb_recall = recall_score(test_data['sentiment'], nb_predictions, average='weighted')
    nb_f1 = f1_score(test_data['sentiment'], nb_predictions, average='weighted')

    print("Naive Bayes Model Evaluation on Test Data:")
    print(classification_report(test_data['sentiment'], nb_predictions))
    print("Accuracy:", nb_accuracy)
    print("Precision:", nb_precision)
    print("Recall:", nb_recall)
    print("F1 Score:", nb_f1)

    # distilBERT model
    train_dataset = Dataset.from_pandas(train_data[['text', 'sentiment']])
    test_dataset = Dataset.from_pandas(test_data[['text', 'sentiment']])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("sentiment", "labels")
    test_dataset = test_dataset.rename_column("sentiment", "labels")

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,  # Reduced number of epochs
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,  # Reduced warmup steps
        weight_decay=0.01,
        logging_dir='./logs',
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("DistilBERT Model Evaluation:")
    bert_eval_result = trainer.evaluate()

    # distilBERT predictions
    bert_test_predictions = trainer.predict(test_dataset)
    bert_preds = torch.argmax(torch.tensor(bert_test_predictions.predictions), dim=1)

    # ensure test_data has 'labels' for comparison
    test_data['labels'] = test_data['sentiment']

    bert_accuracy = accuracy_score(test_data['labels'], bert_preds)
    bert_precision = precision_score(test_data['labels'], bert_preds, average='weighted')
    bert_recall = recall_score(test_data['labels'], bert_preds, average='weighted')
    bert_f1 = f1_score(test_data['labels'], bert_preds, average='weighted')

    print("DistilBERT Model Evaluation on Test Data:")
    print(classification_report(test_data['labels'], bert_preds))
    print("Accuracy:", bert_accuracy)
    print("Precision:", bert_precision)
    print("Recall:", bert_recall)
    print("F1 Score:", bert_f1)

    # visualization
    metrics = ['accuracy', 'precision', 'recall', 'f1 score']
    nb_scores = [nb_accuracy, nb_precision, nb_recall, nb_f1]
    bert_scores = [bert_accuracy, bert_precision, bert_recall, bert_f1]

    x = range(len(metrics))

    plt.figure(figsize=(10, 5))
    plt.bar(x, nb_scores, width=0.4, label='Naive Bayes', align='center')
    plt.bar(x, bert_scores, width=0.4, label='DistilBERT', align='edge')
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Model Comparison: Naive Bayes vs. DistilBERT')
    plt.xticks(x, metrics)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
