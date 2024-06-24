#Group 5, Team members:  Jose Augila, Marcos Portillo, Ben Stewart and Allison Rohrer, Course# AIT526, Date: 6-23-2024
#Question answer system

#explanation and algorithm
#This system will take who, what, where and when questions and search Wikipedia to find the answers.
#For example, if you ask who is George Washington, the system will respond with George Washtington's most relevant titles
#and accomplishments.  For where questions, the system will provide a sentence starting with the place and its location.  
#for what questions, the system will provide a five setence summary of the item.  for when questions, the system 
#will provide a date of the event and the system will also provide birthdays. 
 
#The system extracts a subject and question type from user input, and then uses the subject to find a wikipedia summary. 
#once the wikipedia summary is found, the question type is used to pull the answer from the wikipedia summary.  For example,
#if the question is when was George Washington born?  The system will flag that it is a when question and more specifically a birthday
#question so it will pull the birthdate from the summary provided and construct an answer appropriate for a birthday question.


import wikipedia
import sys
import logging
import spacy
import re
from nltk import sent_tokenize


# initialize spaCy
nlp = spacy.load('en_core_web_sm')

# define answer patterns for different types of questions
patterns = {
    'Who': [
        r"{} is", r"{} was", r"{}'s full name is", r"{}'s full name was",
        r"{} (is|was) (a|an|the)", r"{} (is|was) known as", r"{} (is|was) famous for"
    ],
    'What': [
        r"{} is", r"{} was", r"{} refers to", r"{} can be defined as", r"{} can be described as",
        r"{} (is|was) (a|an|the)", r"{} (is|was) used for", r"{} (is|was) known for", r"{} (is|was) characterized by"
    ],
    'When': [
        r"{} was born on", r"{} was born", r"{} was born in", r"{} was founded on", r"{} was established on",
        r"{} happened on", r"born on {} on", r"born on {} in", r"born on {}", r"{} was",
        r"{}'s birthdate is", r"{}'s birthday is", r"The birthdate of {} is", r"The birthday of {} is",
        r"{} came into the world on", r"{}'s date of birth is", r"{} entered the world on"
    ],
    'Where': [
        r"{} is located in", r"{} is found in", r"{} is situated in", r"{} is in", r"{} is at",
        r"The address of {} is", r"{} is near", r"{} is close to", r"{} is around", r"{} is within",
        r"{} lies in", r"{} sits in", r"{} resides in", r"{} is positioned in", r"{} is placed in"
    ]
}


def compile_patterns():
    return {
        'Who': [re.compile(pattern.format(re.escape('{}')), re.IGNORECASE) for pattern in patterns['Who']],
        'What': [re.compile(pattern.format(re.escape('{}')), re.IGNORECASE) for pattern in patterns['What']],
        'When': [re.compile(pattern.format(re.escape('{}')), re.IGNORECASE) for pattern in patterns['When']],
        'Where': [re.compile(pattern.format(re.escape('{}')), re.IGNORECASE) for pattern in patterns['Where']]
    }


compiled_patterns = compile_patterns()


DATE_PATTERNS = [
    re.compile(r"\((\w+ \d{1,2}, \d{4})\s*[-–]\s*"),
    re.compile(r"\((\d{1,2} \w+ \d{4})\s*[-–]\s*"),
    re.compile(r"born\s+on\s+(\w+ \d{1,2},? \d{4})"),
    re.compile(r"(\d{1,2}\s+\w+\s+\d{4})"),
    re.compile(r"(\w+\s+\d{1,2},?\s+\d{4})")
]

# Location patterns
LOCATION_PATTERNS = [
    re.compile(r'\b(?:located|situated|found|based)\s+in\s+([^.]+)'),
    re.compile(r'in\s+([^.]+)(?:\s+(?:city|state|country|region))?'),
    re.compile(r'(?:at|near)\s+([^.]+)')
]

#functions to search summaries extract dates and locations using date and location patterns
def extract_date(text):
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def extract_location(text):
    for pattern in LOCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None

def check_birth_date_format(summary, subject, question):
    if 'born' in question.lower():
        date = extract_date(summary)
        if date:
            return f"{subject} was born on {date}."
    return None


# functions to return dates, birthdays and locations  using extract date and location functions
def check_date_pattern(summary, subject):
    date = extract_date(summary)
    if date:
        return f"{subject} is associated with the date {date}."
    return None


def check_location_pattern(summary, subject):
    location = extract_location(summary)
    if location:
        return f"{subject} is located in {location}."
    return None

#function to create log file to log questions and answers
def setup_logging(logfile):
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

#function to extract five sentence wikipedia summary using subject 
def get_wikipedia_summary(subject):
    try:
        return wikipedia.summary(subject, sentences=5, auto_suggest=False, redirect=True)
    except wikipedia.DisambiguationError as e:
        logging.info(f"Disambiguation error options: {e.options}")
        return wikipedia.summary(e.options[0], sentences=5, auto_suggest=False, redirect=True)
    except wikipedia.PageError as pe:
        logging.info(f"Wikipedia page error: {pe}")
        return None
    except Exception as ex:
        logging.info(f"General error: {ex}")
        return None

#function to find answer in the wikipedia summary using the question type and subject
def find_answer(question_type, subject, question):
    # generate search patterns
    search_patterns = compiled_patterns[question_type]
    logging.info(f"search patterns: {search_patterns}")
    # print(f"search patterns: {search_patterns}")

    # search Wikipedia
    summary = get_wikipedia_summary(subject)
    if not summary:
        logging.info("No summary found in Wikipedia.")
        return None

    logging.info(f"Wikipedia summary: {summary}")
    # check each sentence in the summary for matches
    sentences = summary.split('. ')
    for sentence in sentences:
        for pattern in search_patterns:
            if pattern.search(sentence):
                logging.info(f"Matched sentence: {sentence}")
                return sentence + "."

    # WHEN check birth date for when questions
    if question_type == 'When':
        birth_date_answer = check_birth_date_format(summary, subject.title(), question)
        logging.info(f"Birth date answer: {birth_date_answer}")
        date_answer = check_date_pattern(summary, subject.title())
        logging.info(f"Date answer: {date_answer}")
        if birth_date_answer and date_answer:
            return birth_date_answer, date_answer
        elif birth_date_answer:
            return birth_date_answer
        else:
            return date_answer
    # WHAT Check for a simple answer if the question type is 'What'
    if question_type == 'What':
        return f"{summary}"
    if question_type == 'Who':
        my_regex = "\(.*\)|\s-\s.*"
        sent_tokens = sent_tokenize(summary)
        summary1 = re.sub(my_regex, "", sent_tokens[0])
        return f"{summary1}"
    if question_type == 'Where':
        location_answer = check_location_pattern(summary, subject)
        if location_answer:
            return location_answer

    return None

#extract the question type and subject
def identify_question_type_and_subject(question):
    doc = nlp(question)
    question_type = None
    subject = None

    # identify the question type
    for token in doc:
        if token.text.lower() in ["who", "what", "when", "where"]:
            question_type = token.text.capitalize()
            break

    # extract the subject
    if question_type:
        subject_chunks = [chunk.text for chunk in doc.noun_chunks]
        if subject_chunks:
            subject = subject_chunks[-1]  # take the last noun chunk as the subject
            # Remove common determiners from the subject
            subject = ' '.join([word.text for word in nlp(subject) if word.pos_ != 'DET'])

    logging.info(f"identified question type: {question_type}, subject: {subject}")
    print(f"identified question type: {question_type}, subject: {subject}")
    return question_type, subject

#function for logging 
def log_and_print(message, level='info'):
    print(message)
    log_func = getattr(logging, level, 'info')
    log_func(message)

#main function to pull questions from user and provide responses using earlier functions 
def main():
    if len(sys.argv) != 2:
        print(len(sys.argv))
        print("usage: python qa-system.py <logfile>")
        return

    logfile = sys.argv[1]

    setup_logging(logfile)
    # logging.basicConfig(filename=logfile, level=logging.INFO)

    print(
        "*** This is a QA system by Group 5. It will try to answer questions that start with Who, What, When or Where. Enter 'exit' to leave the program.")

    while True:
        question = input("=?> ")
        if question.lower() == "exit":
            print("Thank you! Goodbye.")
            break

        # logging.info(f"question: {question}")
        # print(f"question: {question}")
        log_and_print(f"Question: {question}")

        # identify question type and subject
        question_type, subject = identify_question_type_and_subject(question.lower())

        if not question_type or not subject:
            log_and_print("I am sorry, I don't know the answer.", 'info')
            # print("I am sorry, I don't know the answer.")
            # logging.info("answer: I am sorry, I don't know the answer.")
            continue

        # find the answer
        answer = find_answer(question_type, subject, question)
        if answer:
            log_and_print(f"Answer => {answer}", 'info')
            # print(f"=> {answer}")
            # logging.info(f"answer: {answer}")
        else:
            log_and_print("I am sorry, I don't know the answer.", 'info')
            # print("I am sorry, I don't know the answer.")
            # logging.info("answer: I am sorry, I don't know the answer.")


if __name__ == "__main__":
    main()
