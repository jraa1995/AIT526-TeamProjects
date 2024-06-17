import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

nltk.download('stopwords')
nltk.download('wordnet')


# custom transformer for text preprocessing using nltk
class NLTKPreprocessor(TransformerMixin):
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def transform(self, X, **transform_params):
        return [self._clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def _clean_text(self, text):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stopwords]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

# load data
data = pd.read_csv('data/all-data.csv', encoding='cp1252', header=None)

# printing first few rows to understand structure
print(data.head())
print(data.columns)

# rename columns based on inspection
data.columns = ['sentiment', 'text']

# display first few rows of the dataset to verify column renaming
print(data.head())

# preprocessing
X = data['text']  # text data is in the 'text' column
y = data['sentiment']  # sentiment labels are in the 'sentiment' column

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a pipeline that includes text preprocessing, vectorization, and the classifier
pipeline = Pipeline([
    ('preprocess', NLTKPreprocessor()),
    ('vectorize', TfidfVectorizer()),
    ('classify', MultinomialNB())
])

# perform hyperparameter tuning
param_grid = {
    'vectorize__max_df': [0.8, 0.9, 1.0],
    'vectorize__ngram_range': [(1, 1), (1, 2)],
    'classify__alpha': [0.01, 0.1, 1.0]
}

# set n_jobs=1 to avoid serialization issues
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=1, verbose=1)
grid_search.fit(X_train, y_train)

# best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# predict and evaluate on the test set
y_pred = grid_search.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# function to compare model prediction and actual sentiment for a single example
def compare_results(text, model):
    model_pred = model.predict([text])[0]
    print(f"Text: {text}")
    print(f"Model Prediction: {model_pred}")


# compare for a few samples
for text in X_test.sample(5):
    compare_results(text, grid_search)
