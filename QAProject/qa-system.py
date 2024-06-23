import wikipedia
import sys
import logging
import spacy
import re
from nltk import sent_tokenize

#notes - need to add additional date formats, need to make sure there is way to distinguish birthday's from dates,  need to add location regex

# initialize spaCy
nlp = spacy.load('en_core_web_sm')

# define answer patterns for different types of questions
patterns = {
    'Who': [
        r"{} is", r"{} was", r"{}'s full name is", r"{}'s full name was"
    ],
    'What': [
        r"{} is", r"{} was", r"{} refers to", r"{} can be defined as"
    ],
    'When': [
        r"{} was born on", r"{} was born", r"{} was born in", r"{} was founded on", r"{} was established on",
        r"{} happened on",
        r"born on {} on", r"born on {} in", r"born on {}", r"{} was"
    ],
    'Where': [
        r"{} is located in", r"{} is found in", r"{} is situated in", r"{} is in", r"{} is at"
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

BIRTH_DATE_PATTERN = re.compile(r"\((\w+ \d{1,2}, \d{4}) â")

DATE_PATTERN = re.compile(r"\((\d{1,2} \w+ \d{4}) â")
DATEPATTERN1 = re.compile(r'\d{1,2} \w+ \d{4}')
datepattern2 = re.compile(r"\((\w+ \d{1,2}, \d{4}) â")

locationpattern = re.compile(r'\blocated.*\b')
locationpattern1 = re.compile(r'in [^.]*\.')

def check_birth_date_format(summary, subject, question):
    if 'born'in question:
        match = BIRTH_DATE_PATTERN.search(summary)
        if match:
            birth_date = match.group(1)
            return f"{subject} was born on {birth_date}."
    return None

def check_date_pattern(summary, subject):
   match = DATE_PATTERN.search(summary)
   match1 = DATEPATTERN1.search(summary)
   match2 = datepattern2.search(summary)
   if match1:
       date = match1.group(0)
       return f"{subject} started on {date}."
   if match:
       date = match.group(1)
       return f"{subject} started on {date}."
   if match2:
       date = match2.group(1)
       return f"{subject} started on {date}."
   return None

def check_location_pattern(summary, subject):
    match = locationpattern.search(summary)
    match1 = locationpattern1.search(summary)
    if match:
        location = match.group(0)
        return f"{subject} is {location}." 
    if match1:
        location = match1.group(0)
        return f"{subject} is {location}"
    return None

def setup_logging(logfile):
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

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

def find_answer(question_type, subject, question):
    # generate search patterns
    search_patterns = compiled_patterns[question_type]
    logging.info(f"search patterns: {search_patterns}")
    #print(f"search patterns: {search_patterns}")

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
        birth_date_answer = check_birth_date_format(summary, subject, question)
        date_answer = check_date_pattern(summary, subject)
        if birth_date_answer:
            return birth_date_answer
        if date_answer:
            return date_answer
    # WHAT Check for a simple answer if the question type is 'What'
    if question_type == 'What':
        return f"{summary}"
    if question_type == 'Who':
        my_regex = "\(.*\)|\s-\s.*"
        sent_tokens = sent_tokenize(summary)
        summary1 = re.sub(my_regex, "", sent_tokens[0])
        return f"{summary1}"
    if question_type == 'Where' :
        location_answer = check_location_pattern(summary, subject)
        if location_answer:
            return location_answer

    return None


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

def log_and_print(message, level='info'):
    print(message)
    log_func = getattr(logging, level, 'info')
    log_func(message)

def main():
    if len(sys.argv) != 2:
        print(len(sys.argv))
        print("usage: python qa-system.py <logfile>")
        return

    logfile = sys.argv[1]
  
    setup_logging(logfile)
    #logging.basicConfig(filename=logfile, level=logging.INFO)

    print("*** This is a QA system by Group 5. It will try to answer questions that start with Who, What, When or Where. Enter 'exit' to leave the program.")

    while True:
        question = input("=?> ")
        if question.lower() == "exit":
            print("Thank you! Goodbye.")
            break

        #logging.info(f"question: {question}")
        #print(f"question: {question}")
        log_and_print(f"Question: {question}")

        # identify question type and subject
        question_type, subject = identify_question_type_and_subject(question)

        if not question_type or not subject:
            log_and_print("I am sorry, I don't know the answer.", 'info')
            #print("I am sorry, I don't know the answer.")
            #logging.info("answer: I am sorry, I don't know the answer.")
            continue

        # find the answer
        answer = find_answer(question_type, subject, question)
        if answer:
            log_and_print(f"Answer => {answer}", 'info')
            #print(f"=> {answer}")
            #logging.info(f"answer: {answer}")
        else:
            log_and_print("I am sorry, I don't know the answer.", 'info')
            #print("I am sorry, I don't know the answer.")
            #logging.info("answer: I am sorry, I don't know the answer.")


if __name__ == "__main__":
    main()
