# Naive Bayes Sentiment Classifier

This repository contains a Naive Bayes classifier for sentiment classification of airline reviews. The classifier is built from scratch using Python and processes the reviews to determine if they are positive or negative.

## Project Structure

SENTIMENTCLASS
├── __MACOSX
├── Classifier Project
├── tweet (contains train and test subdirectories)
├── venv (virtual environment)
├── classifier.py
├── results_stemming_False_binary.txt
├── results_stemming_False_frequency.txt
├── results_stemming_True_binary.txt
├── results_stemming_True_frequency.txt
├── tweet.zip
├── .gitignore
└── README.md

markdown
Copy code

## Setup

### Prerequisites

- Python 3.11
- `nltk` library
- `beautifulsoup4` library
- `emoji` library

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/jraa1995/AIT526-TeamProjects.git
    cd AIT526-TeamProjects
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

### Project Structure

- `classifier.py`: The main script containing the Naive Bayes classifier.
- `results_stemming_*.txt`: Result files containing the output of the classifier with different configurations.
- `tweet.zip`: The dataset containing training and testing data.
- `.gitignore`: File to ignore unnecessary files and directories.
- `README.md`: Project documentation.

## Usage

1. **Unzip the `tweet.zip` file**:
    ```bash
    unzip tweet.zip
    ```

2. **Run the classifier**:
    ```bash
    python Classifier\ Project/classifier.py
    ```

## Results

The results of the classifier are saved in the following files:
- `results_stemming_False_binary.txt`
- `results_stemming_False_frequency.txt`
- `results_stemming_True_binary.txt`
- `results_stemming_True_frequency.txt`

Each file contains the accuracy, confusion matrix, and performance metrics of the classifier with different configurations.

