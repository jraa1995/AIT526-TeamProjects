import wikipedia
import sys
import logging
import spacy
import re

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
        r"born on {} on", r"born on {} in", r"born on {}"
    ],
    'Where': [
        r"{} is located in", r"{} is found in", r"{} is situated in", r"{} is in", r"{} is at"
    ]
}


def find_answer(question_type, subject):
    # generate search patterns
    search_patterns = [re.compile(pattern.format(re.escape(subject)), re.IGNORECASE) for pattern in
                       patterns[question_type]]
    logging.info(f"search patterns: {search_patterns}")
    print(f"search patterns: {search_patterns}")

    # search Wikipedia
    try:
        summary = wikipedia.summary(subject, sentences=5, auto_suggest=False, redirect=True)
        logging.info(f"wikipedia summary: {summary}")
        print(f"wikipedia summary: {summary}")
    except wikipedia.DisambiguationError as e:
        logging.info(f"disambiguation error options: {e.options}")
        print(f"disambiguation error options: {e.options}")
        try:
            summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False, redirect=True)
            logging.info(f"wikipedia summary after disambiguation: {summary}")
            print(f"wikipedia summary after disambiguation: {summary}")
        except Exception as ex:
            logging.info(f"error after disambiguation: {ex}")
            print(f"error after disambiguation: {ex}")
            return None
    except wikipedia.PageError as pe:
        logging.info(f"wikipedia page error: {pe}")
        print(f"wikipedia page error: {pe}")
        return None
    except Exception as ex:
        logging.info(f"general error: {ex}")
        print(f"general error: {ex}")
        return None

    # check each sentence in the summary for matches
    sentences = summary.split('. ')
    for sentence in sentences:
        for pattern in search_patterns:
            if pattern.search(sentence):
                logging.info(f"matched sentence: {sentence}")
                print(f"matched sentence: {sentence}")
                return sentence + "."

    # check birth date format
    birth_date_pattern = re.compile(r"\((\w+ \d{1,2}, \d{4}) –")
    match = birth_date_pattern.search(summary)
    if match:
        birth_date = match.group(1)
        return f"{subject} was born on {birth_date}."

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

    logging.info(f"identified question type: {question_type}, subject: {subject}")
    print(f"identified question type: {question_type}, subject: {subject}")
    return question_type, subject


def main():
    if len(sys.argv) != 2:
        print("usage: python qa-system.py <logfile>")
        return

    logfile = sys.argv[1]
    logging.basicConfig(filename=logfile, level=logging.INFO)

    print(
        "*** This is a QA system by Group 5. It will try to answer questions that start with Who, What, When or Where. Enter 'exit' to leave the program.")

    while True:
        question = input("=?> ")
        if question.lower() == "exit":
            print("Thank you! Goodbye.")
            break

        logging.info(f"question: {question}")
        print(f"question: {question}")

        # identify question type and subject
        question_type, subject = identify_question_type_and_subject(question)

        if not question_type or not subject:
            print("I am sorry, I don't know the answer.")
            logging.info("answer: I am sorry, I don't know the answer.")
            continue

        # find the answer
        answer = find_answer(question_type, subject)
        if answer:
            print(f"=> {answer}")
            logging.info(f"answer: {answer}")
        else:
            print("I am sorry, I don't know the answer.")
            logging.info("answer: I am sorry, I don't know the answer.")


if __name__ == "__main__":
    main()
