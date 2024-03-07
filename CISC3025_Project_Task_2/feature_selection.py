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


def feature_selection(input_file, threshold=None, output_file=None):
    # -------------- 1. Preparation ---------------#
    # 1.1 Initialize Tokenizer
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\s]+', gaps=True)

    # 1.2 Initialize Class Frequency Vector
    word_freqs_for_each_class_arr = [0, 0, 0, 0, 0]

    # ---------- 2. Read & Extract Data ------------#
    # Open input file, read file line by line.
    # Make sure to run count_word.py before running this code.
    _, selected_features = __funcs__.extract_data_from_txt(input_file, tokenizer, max_threshold=threshold)

    # -------------- 3. Post Process ---------------#
    # Summarize word frequencies into class-wise word frequencies.
    # Given rows in selected_features like: ['apple',[10,44,13,1918,809]]
    for instance in selected_features:
        cur_data_set = instance[1]                  # Frequencies of a word.
        for index in range(0, len(cur_data_set)):   # Accumulate freq to class-wise array.
            word_freqs_for_each_class_arr[index] += cur_data_set[index]

    # --------------- 4. Write Data ----------------#
    if output_file is None:
        return word_freqs_for_each_class_arr, selected_features

    __funcs__.write_data_to_txt(word_freqs_for_each_class_arr, selected_features, output_file)

    return word_freqs_for_each_class_arr, selected_features


feature_selection('./output/word_count.txt',  10000, './output/word_dict.txt')
