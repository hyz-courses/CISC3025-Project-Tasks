"""
CISC3025-Project-Task-02 Requirement 2: Feature Selection.
Select the first 10,000 features ordered by frequency of occurrence extracted from the training data.
"""

import json
import re
import nltk
import pandas as pd
import numpy as np
from collections import Counter

# local imports
import __funcs__
from __funcs__ import settings

input_file = './output/word_count.txt'
output_file = './output/word_dict.txt'


def feature_selection(input_file_path, threshold=None, output_file_path=None):
    """
    Select the first 10,000 features ordered by frequency of occurrence extracted from the training.
    """

    ''' ---------------- 1. Data Extraction ------------------- '''
    # 1.1 Initialize Tokenizer
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\s]+', gaps=True)

    # 1.2 Initialize Class token-frequency Vector
    word_freqs_for_each_class_arr = [0, 0, 0, 0, 0]

    # 1.3 Extract data from token-frequency matrix and filter.
    # list of ['word', [f1, f2, f3, f4, f5]]
    _, train_word_freqs_list = __funcs__.extract_data_from_txt(input_file_path, tokenizer, max_threshold=threshold)

    ''' ---------------- 2. Post Processing ------------------- '''
    # Summarize word frequencies into class-wise word frequencies.
    # Given rows in selected_features like: ['apple',[10,44,13,1918,809]]
    for instance in train_word_freqs_list:
        cur_data_set = instance[1]                  # Frequencies of a word.
        for index in range(0, len(cur_data_set)):   # Accumulate freq to class-wise array.
            word_freqs_for_each_class_arr[index] += cur_data_set[index]

    # --------------- 4. Write Data ----------------#
    if output_file_path is None:
        return word_freqs_for_each_class_arr, train_word_freqs_list

    __funcs__.write_data_to_txt(word_freqs_for_each_class_arr, train_word_freqs_list, output_file_path)

    return word_freqs_for_each_class_arr, train_word_freqs_list


#feature_selection(input_file,  10000, output_file)
