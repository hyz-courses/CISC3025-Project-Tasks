import json
import re
import nltk
import pandas as pd
import numpy as np
from collections import Counter

settings = {
    "PRINT_PROCESS": True,
    "WRITE_FILES": True,
}

# ANSI Colors: For better distinguishable in console.
color = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "default": "\033[0m"
}

# Map class name to index number.
class_map = {
    'crude': 0,
    'grain': 1,
    'money-fx': 2,
    'acq': 3,
    'earn': 4
}
classes = ['crude', 'grain', 'money-fx', 'acq', 'earn']


def extract_data_from_txt(input_file, tokenizer, max_threshold=None, include_first_line=False, convert_to_int=True):
    """
    :param input_file: Path to the input txt file;
    :param tokenizer: Tokenizer object;
    :param max_threshold: Threshold of selecting input txt lines. Default None;
    :param include_first_line: Whether to read the first line of the input txt file.
    :param convert_to_int: Convert to integer.

    This function extracts data from txt file and turn them into a list of pairs.
    The first element of the pair is the word in the vocabulary;
    The second element of the pair is a length-5 integer list of word related data.
    Primary key is word, index is classes.
    """

    selected_features = []
    first_line = ""

    with open(input_file, 'r') as i_file:
        for line_number, line in enumerate(i_file, start=1):
            # Ignore the first row, since they are doc freqs of classes.
            if line_number == 1:
                first_line = tokenizer.tokenize(line) if include_first_line else None
                continue

            # Never exceed the given threshold.
            if max_threshold is not None and line_number > max_threshold + 1:
                break

            # 2.1 Tokenize cur line.
            cur_line = line.strip()     # example: "apple 10 44 13 1918 809"
            cur_data_set = tokenizer.tokenize(cur_line)     # example: ['apple', '10', '44', '13', '1919','809']

            # 2.2 Separate words and frequencies.
            cur_word = cur_data_set[0]          # Current word
            cur_data_set = cur_data_set[1:]     # Slice the first word, leaving the frequencies.
            if convert_to_int:
                cur_data_set = [int(num) for num in cur_data_set]   # Convert string to int

            # Summarize
            # Sample row: ['apple',[10,44,13,1918,809]]
            selected_features.append((cur_word, cur_data_set))

        i_file.close()
    return first_line, selected_features


def write_data_to_txt(first_line, word_datas_tuples, output_file):
    """
    :param first_line: A length-5 list of data.
    :param word_datas_tuples: A pair of word-data. Data is a length-5 list of numeric data assigned to a word.
    :param output_file: File path to the output file.

    This function writes data into the output file.
    """

    # First line
    first_line_str = ""
    for num in first_line:
        first_line_str += str(num) + " "
    first_line_str += "\n"

    # Word Features
    with open(output_file, 'w') as o_file:
        o_file.write(first_line_str) if settings['WRITE_FILES'] else None
        print(first_line_str) if settings['PRINT_PROCESS'] else None

        for index, instance in enumerate(word_datas_tuples):
            cur_word = instance[0]      # Cur word
            cur_data_set = instance[1]  # Cur word's data in all classes

            # Initialize write string
            cur_word_data_str = ""
            cur_word_data_str += cur_word + " "

            # Write each data in instance[1] into string
            for num in cur_data_set:
                cur_word_data_str += str(num) + " "
            cur_word_data_str += "\n"

            # Finally, write the processed string into file.
            o_file.write(cur_word_data_str) if settings['WRITE_FILES'] else None

            if not settings['PRINT_PROCESS']:
                continue

            print(str(index+1), end=" ")
            print(cur_word_data_str)

        o_file.close()


def extract_data_from_json(input_file_path, tokenizer, include_id=False):
    """
    :param input_file_path: Path to the input .json file.
    :param tokenizer: Tokenizer that uses the compiled delimiter rules to tokenize long sentences.
    :param include_id: Whether to include file id.
    """

    # ------------ 1. Preparation ------------- #
    # 1.1 Initialize Output
    class_freqs = [0, 0, 0, 0, 0]       # Document frequencies for each class.
    t_class_sentence_list = []          # Instance example: [class_name, ['token1', 'token2','token1']]
    t_class_dict_list = []              # Instance example: [class_name, {'token1':2, 'token2':1}]
    stemmer = nltk.PorterStemmer()

    # 1.2 Open file, read json data.
    with open(input_file_path, 'r') as f:
        class_and_sentence_list = json.load(f)

    # ------------- 2. Extraction -------------#
    for instance in class_and_sentence_list:
        # For each instance of form [class_name, "token1 token2 token3 ..."]:

        # 2.0 Record the file ID
        cur_file_id = instance[0]

        # 2.1 Record the class of this instance and accumulate doc freq.
        cur_class = instance[1]
        class_freqs[class_map[cur_class]] += 1

        # 2.2 Record current un-tokenized sentence.
        cur_sentence = instance[2]

        # 2.3 Pre-process the sentence for further tokenization.
        # 2.3.1 Replace common delimiters by a single space and eliminate some useless characters.
        # Remove double-commas, abbreviation marks, brackets, and continuous hyphens.
        cur_sentence = re.sub(r'\"|\.\.+|\(|\)|\s--+\s|(?<=[A-Za-z])/|&[a-z]+;|>', ' ', cur_sentence)
        # 2.3.2 Remove period, comma, question mark and exclamation mark followed by a  space as common delimiters.
        # Leave abbreviations alone, like U.S. or U.K.
        cur_sentence = re.sub(r'(?<![A-Z])([.,?!"]\s+)', ' ', cur_sentence)

        # 2.4 Tokenize current sentence.
        _cur_token_array = tokenizer.tokenize(cur_sentence)
        cur_token_array = []

        # 2.5 Post-processing. Remove remaining cases where there's a punctuation mark at the end of a word.
        for word in _cur_token_array:
            word = word.rstrip(',?!"-')
            word = word.lower()
            cur_token_array.append(word) if word != "-" or "" else None

        cur_token_array = [stemmer.stem(word) for word in cur_token_array]

        # 2.6 Summarize Data
        # 2.6.1 Array Part
        t_class_sentence = [cur_class, cur_token_array]
        if include_id is True:
            t_class_sentence.append(cur_file_id)
        t_class_sentence_list.append(t_class_sentence)

        # 2.6.2 Frequency part
        cur_freq_dict = Counter(cur_token_array)
        cur_freq_dict = dict(cur_freq_dict)
        t_class_dict = [cur_class, cur_freq_dict]
        if include_id is True:
            t_class_dict.append(cur_file_id)
        t_class_dict_list.append(t_class_dict)

    return class_freqs, t_class_sentence_list, t_class_dict_list
