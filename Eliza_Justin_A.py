# eliza.py
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:39:02 2024

@author: ajroh
"""
from nltk.tokenize import word_tokenize
import re

# Constants
NAME_PROMPT = "Hi, I'm Eliza, a psychotherapist. What is your name?"
REPEAT_NAME_PROMPT = "Can you repeat your name?"
INITIAL_PROMPT = "{name}, what is on your mind today?"
DEFAULT_RESPONSE = "Tell me more."
APOLOGY_RESPONSE = "There is no need to apologize."
UNCLEAR_RESPONSE = "I didn't quite understand, can you say that another way?"

def main():
    """
    main function to start the eliza program
    greets the user, gets their name, and initiates a dialogue loop
    """
    print(NAME_PROMPT)
    name = get_user_name()
    print(INITIAL_PROMPT.format(name=name))
    input_text = input('You: ')

    while input_text.strip() != '':
        response = generate_response(input_text, name)
        print(response)
        input_text = input('You: ')


def get_user_name():
    """
    function to get the user's name
    handles different ways the user might introduce themselves

    returns:
    str: user's name
    """
    while True:
        name = input("You: ")
        tokens = word_tokenize(name)

        if len(tokens) == 1:
            return tokens[0]
        elif tokens[:3] == ['my', 'name', 'is']:
            return tokens[3]
        else:
            print(REPEAT_NAME_PROMPT)


def generate_response(input_text, name):
    """
    generates a response based on the user's input text and name.

    parameters:
    input_text (str): user's input text
    name (str): user's name

    returns:
    str: eliza's response
    """
    input_text = input_text.lower()  # convert input to lowercase for case-insensitive matching

    # respond to statements with "feel" or "crave"
    if re.search(r'\bfeel\b', input_text):
        return re.sub(r'i (feel|crave) (.+)', r'Tell me more about those \1ings', input_text, flags=re.IGNORECASE)
    # respond to statements with "want"
    elif re.search(r'\bwant\b', input_text):
        return f"{name}, why do you want that?"
    # respond to explanations with "because"
    elif re.search(r'\bbecause\b', input_text):
        return "Is that the real reason?"
    # respond to apologies
    elif re.search(r'\bsorry\b', input_text):
        return "There is no need to apologize."
    # respond to generalizations with "always" or "never"
    elif re.search(r'\balways\b|\bnever\b', input_text):
        return "Can you be more specific?"
    # respond to statements with "can't" or "cannot"
    elif re.search(r'\bcan\'t\b|\bcannot\b', input_text):
        return "What makes you think you can't?"
    # respond to statements with "think"
    elif re.search(r'\bthink\b', input_text):
        return "Do you really think so?"
    # respond to statements with "I am"
    elif re.search(r'\bam\b', input_text):
        return re.sub(r'\bi am (.+)', r'How long have you been \1?', input_text, flags=re.IGNORECASE)
    # respond to questions with "Are you"
    elif re.search(r'\bare you\b', input_text):
        return re.sub(r'\bare you (.+)', r'Why does it matter whether I am \1?', input_text, flags=re.IGNORECASE)
    # respond to statements with "you are"
    elif re.search(r'\byou are\b', input_text):
        return re.sub(r'\byou are (.+)', r'What makes you think I am \1?', input_text, flags=re.IGNORECASE)
    # respond to statements with "I need"
    elif re.search(r'\bi need\b', input_text):
        return re.sub(r'\bi need (.+)', r'Why do you need \1?', input_text, flags=re.IGNORECASE)
    # respond to inquiries about Eliza's state
    elif re.search(r'\bhow are you\b', input_text):
        return "I'm just a program, but I'm here to help you."
    # respond to mentions of family
    elif re.search(r'\bmother\b|\bfather\b|\bfamily\b', input_text):
        return "Tell me more about your family."
    # respond to greetings
    elif re.search(r'\bhello\b|\bhi\b|\bhey\b', input_text):
        return f"Hello {name}, how can I help you today?"
    # respond to requests for help
    elif re.search(r'\bhelp\b', input_text):
        return "I'm here to help you. What seems to be the problem?"
    # respond to mentions of relationships
    elif re.search(r'\bmy (wife|husband|boyfriend|girlfriend)\b', input_text):
        relationship = re.search(r'\bmy (wife|husband|boyfriend|girlfriend)\b', input_text).group(1)
        return f"Tell me more about your {relationship}."
    # respond to statements with "I'm"
    elif re.search(r'\bi\'m (.+)', input_text):
        return re.sub(r'\bi\'m (.+)', r'How does being \1 make you feel?', input_text, flags=re.IGNORECASE)
    # respond to statements with "you [verb] me"
    elif re.search(r'\byou (.+) me\b', input_text):
        return re.sub(r'\byou (.+) me\b', r'What makes you think I \1 you?', input_text, flags=re.IGNORECASE)
    # respond to statements with "I [verb] you"
    elif re.search(r'\bi (.+) you\b', input_text):
        return re.sub(r'\bi (.+) you\b', r'Why do you \1 me?', input_text, flags=re.IGNORECASE)
    # respond to statements with "you're"
    elif re.search(r'\byou\'re (.+)\b', input_text):
        return re.sub(r'\byou\'re (.+)\b', r'Why do you think I am \1?', input_text, flags=re.IGNORECASE)
    # respond to statements with "I'm [state] because"
    elif re.search(r'\bi\'m (.+) because\b', input_text):
        return re.sub(r'\bi\'m (.+) because\b', r'Do you think being \1 is the real reason?', input_text,
                      flags=re.IGNORECASE)
    # respond to statements with "no one"
    elif re.search(r'\bno one\b', input_text):
        return "Why do you feel that no one " + input_text.split('no one', 1)[1] + "?"
    # respond to statements with "everyone"
    elif re.search(r'\beveryone\b', input_text):
        return "Can you think of anyone in particular?"
    # respond to affirmations
    elif re.search(r'\byes\b', input_text):
        return "I see. Can you elaborate on that?"
    # respond to negations
    elif re.search(r'\bno\b', input_text):
        return "Why not?"
    # respond to completions
    elif re.search(r'\bdone\b', input_text):
        return "How do you feel about that?"
    # respond to conditional statements
    elif re.search(r'\bif\b', input_text):
        return "Do you think that's likely?"
    # respond to "what" questions
    elif re.search(r'\bwhat\b', input_text):
        return "Why do you ask?"
    # respond to "how" questions
    elif re.search(r'\bhow\b', input_text):
        return "What are your thoughts on that?"
    # respond to "why" questions
    elif re.search(r'\bwhy\b', input_text):
        return "Why do you think that is?"
    # respond to "when" questions
    elif re.search(r'\bwhen\b', input_text):
        return "Does that time have any special significance?"
    # respond to "where" questions
    elif re.search(r'\bwhere\b', input_text):
        return "Why do you ask about that place?"
    # respond to gibberish or non-alphabetic input
    elif not any(word.isalpha() for word in word_tokenize(input_text)):
        return UNCLEAR_RESPONSE
    # default response for unrecognized patterns
    else:
        return DEFAULT_RESPONSE


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
