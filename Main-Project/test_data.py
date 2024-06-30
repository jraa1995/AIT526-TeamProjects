import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# load the datasets
all_news_sentiments = pd.read_csv('Main-Project/all_news_sentiments.csv')

# convert dates to datetime and remove timezone if present
all_news_sentiments['publishedDate'] = pd.to_datetime(all_news_sentiments['publishedDate']).dt.tz_localize(None)

# remove duplicates based on title
all_news_sentiments = all_news_sentiments.drop_duplicates(subset=['title'])

# map sentiment labels to integers
sentiment_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
all_news_sentiments['sentiment'] = all_news_sentiments['sentiment'].map(sentiment_mapping)

# check dataset size
print(f"Total dataset size: {all_news_sentiments.shape[0]} rows")

# split the data into training and test sets
train_data, test_data = train_test_split(all_news_sentiments, test_size=0.2, random_state=42, stratify=all_news_sentiments['sentiment'])

# check for overlap between training and test data
train_titles = set(train_data['title'])
test_titles = set(test_data['title'])
overlap = train_titles.intersection(test_titles)
print(f"Overlap between training and test data: {len(overlap)} articles")
assert len(overlap) == 0, "Overlap detected between training and test data!"

# check the distribution of sentiment labels
print("Training data sentiment distribution:")
print(train_data['sentiment'].value_counts())
print("\nTest data sentiment distribution:")
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
print("Naive Bayes Model Cross-Validation Results:")
print(cv_df.mean())

# final training and evaluation on the test set
nb_model.fit(X_train_sm, y_train_sm)
X_test = vectorizer.transform(test_data['text'])
nb_predictions = nb_model.predict(X_test)

print("Naive Bayes Model Evaluation on Test Data:")
print(classification_report(test_data['sentiment'], nb_predictions))
print("Accuracy:", accuracy_score(test_data['sentiment'], nb_predictions))
print("Precision:", precision_score(test_data['sentiment'], nb_predictions, average='weighted'))
print("Recall:", recall_score(test_data['sentiment'], nb_predictions, average='weighted'))
print("F1 Score:", f1_score(test_data['sentiment'], nb_predictions, average='weighted'))

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

print("BERT Model Evaluation:")
bert_eval_result = trainer.evaluate()

# bert predictions
bert_test_predictions = trainer.predict(test_dataset)
bert_preds = torch.argmax(torch.tensor(bert_test_predictions.predictions), dim=1)

# bert evaluation
print("BERT Model Evaluation on Test Data:")
print(classification_report(test_data['sentiment'], bert_preds))
print("Accuracy:", accuracy_score(test_data['sentiment'], bert_preds))
print("Precision:", precision_score(test_data['sentiment'], bert_preds, average='weighted'))
print("Recall:", recall_score(test_data['sentiment'], bert_preds, average='weighted'))
print("F1 Score:", f1_score(test_data['sentiment'], bert_preds, average='weighted'))
