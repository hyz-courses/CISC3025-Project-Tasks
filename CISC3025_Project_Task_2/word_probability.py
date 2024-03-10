'''
CISC3025-Project-Task-02 Requirement 3: Word Probability
'''

import json
import re
import nltk
import math
import pandas as pd
import numpy as np
from collections import Counter

# local imports
import __funcs__

settings = __funcs__.settings


def word_probability(input_files, output_file=None, add_alpha=0, precision=None, use_log=False):
    # -----------------1.Preparation----------------- #
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\s]+', gaps=True)

    # 1.1 Num of documents in each class.
    doc_freqs_for_each_class_arr, _ = __funcs__.extract_data_from_txt(
        input_files[0],
        tokenizer,
        max_threshold=1,
        include_first_line=True)

    # 1.2 Num of words in each class, and a list of word-frequencies pairs.
    # Sample row of the list: ['word1', ['10','44','10','1918','809']]
    word_freqs_for_each_class_arr, feature_words = __funcs__.extract_data_from_txt(
        input_files[1],
        tokenizer,
        include_first_line=True)

    # -----------------2.Prob of each class----------------- #
    # 2.1 Convert number string into integer. Get vector of doc frequencies in each class.
    doc_freqs_for_each_class_arr = [int(num_str) for num_str in doc_freqs_for_each_class_arr]

    # 2.2 Get number of documents.
    num_of_docs = sum(doc_freqs_for_each_class_arr)

    # 2.3 Document probabilities in each class. (Portion of each class)
    doc_probs_for_each_class_arr = [num/num_of_docs for num in doc_freqs_for_each_class_arr]

    # 2.4 If required, convert to log space.
    if use_log:
        doc_probs_for_each_class_arr = [-math.log(num) for num in doc_probs_for_each_class_arr]

    # -----------------3. Prob of each word----------------- #
    # 3.1 Convert number string into integer. Get word frequencies for each class.
    word_freqs_for_each_class_arr = [int(num_str) for num_str in word_freqs_for_each_class_arr]

    # 3.2 Get: A list of word-list pairs.
    # Sample row: ['word1',['0.5','0.4','0.1','0.8','0.02']]. Numeric data is P(word|class)
    feature_word_probs = []
    vocabulary_size = len(feature_words)
    for instance in feature_words:
        # 3.2.1 Extract info
        cur_word = instance[0]          # Current word
        cur_word_freqs = instance[1]    # Current frequencies in each class of this word

        # 3.2.2 Calculate portion of this word in each class.
        cur_word_probs = [
            # Word freq divide by corresponding word num
            (word_freq + add_alpha) / (word_freqs_for_each_class_arr[class_code] + (add_alpha * vocabulary_size))
            for class_code, word_freq in enumerate(cur_word_freqs)
        ]

        if precision is not None:
            cur_word_probs = [round(num, precision) for num in cur_word_probs]

        if use_log:
            cur_word_probs = [-math.log(num) for num in cur_word_probs]

        # 3.2.3 Append word and probs into list
        feature_word_probs.append([cur_word, cur_word_probs])

    # -----------------4. Write Files----------------- #
    if output_file is None:
        return

    __funcs__.write_data_to_txt(doc_probs_for_each_class_arr, feature_word_probs, output_file)


# word_probability(
#     ["./output/word_count.txt", "./output/word_dict.txt"],
#     output_file="./output/word_probability.txt",
#     add_alpha=1,
#     use_log=False
#     # precision=4   # Control digits after dot.
# )
#
# word_probability(
#     ["./output/word_count.txt", "./output/word_dict.txt"],
#     output_file="./temp_output/word_probability_log.txt",
#     add_alpha=1,
#     use_log=True
#     # precision=4   # Control digits after dot.
# )
