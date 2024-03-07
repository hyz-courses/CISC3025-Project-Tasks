'''
CISC3025-Project-Task-02 Requirement 4: Classification
'''

import os
import json
import re
import nltk
import math
import pandas as pd
import numpy as np
from collections import Counter

import __funcs__

custom_settings = {
    "WRITE_DATA": False
}

input_file = './data/test.json'
output_file = './output/classification_result.txt'


def classification(input_file_path, output_file_path):
    # ------------ 1.Preparation -------------- #
    # 1.1 Initialize Tokenizer: Split sentence using space characters.
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\s]+', gaps=True)

    # -------- 2. Preprocess Classifier ---------#
    # 2.1 Retrieve logged and negatived values of all probs.
    # - First param: -log[P(c)] for all c in C. C is the set of all classes.
    # - Second param: List of word-prob pairs. In each pair, let word be w, then probs are -log[P(w|c)] for all c.
    # - Sample row of second param: ['word', ['val1', 'val2', 'val3', 'val4', 'val5']]
    [
        log_neg_doc_probs_for_each_class,        # -log[P(c)]
        _log_neg_word_probs_arr                  # -log[P(w|c)] for all w in V, in str format.
    ] = __funcs__.extract_data_from_txt(
        "./temp_output/word_probability_log.txt",
        tokenizer,
        include_first_line=True,
        convert_to_int=False)

    # 2.2 For all c in C, convert string to float. Get -log[P(c)] for all c in C.
    log_neg_doc_probs_for_each_class = [float(num) for num in log_neg_doc_probs_for_each_class]

    # 2.3 For each word-probs pairs, convert string to float. And calculate joined probs.
    # Get -log[P(w|c)] for all words in all classes.
    log_neg_word_probs_arr = []
    for instance in _log_neg_word_probs_arr:
        # 4.3.1 Basic data.
        cur_word = instance[0]                  # Current Word
        _cur_log_neg_word_probs = instance[1]    # Current Prob in string format

        # 4.3.2 Convert float to string.
        cur_log_neg_word_probs = [float(num) for num in _cur_log_neg_word_probs]     # -log[P(w|c)] for all c
        log_neg_word_probs_arr.append([cur_word, cur_log_neg_word_probs])

    # ------------- 3. Calculate Naive Bayes Prob -------------#
    [
        test_class_freq,
        test_word_sentence_list,
        test_word_dict_list
    ] = __funcs__.extract_data_from_json(input_file_path, tokenizer)

    vocab = set()
    for instance in test_word_dict_list:
        for word in instance[1]:
            vocab.add(word)
    vocab = list(vocab)

    for instance in test_word_sentence_list:
        test_cur_class = instance[0]
        test_cur_word_arr = instance[1]
        print(test_cur_class, end=" ")
        print(test_cur_word_arr)

        for word in test_cur_word_arr:
            # TODO: Index this word in the joined probability matrix.

            # If present, get the list of joined probs

            # If not, set the list to empty_prob_val = 1 / (word freqs for class + vocab size)

            pass


classification(input_file,output_file)