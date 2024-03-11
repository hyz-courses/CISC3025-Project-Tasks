'''
CISC3025-Project-Task-02 Requirement 1: Count Words
'''

import os
import json
import re
import nltk
import pandas as pd
import numpy as np
from collections import Counter

import __funcs__

input_file = './data/train.json'
output_file = './output/word_count.txt'


def count_word(input_file_path, output_file_path):
    """
    :param input_file_path: Path to the input file.
    :param output_file_path: Path to the output file.

    :return tf_matrix: Sorted Token-Frequency matrix in each class.
    """

    ''' ---------------- 1. Data Extraction ------------------- '''
    # 1.1 Tokenizer: Split sentence using space characters.
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\s]+', gaps=True)

    # 1.2 Retrieve processed training data.
    [
        train_class_doc_freqs,          # Doc Freq for each class
        train_class_sentence_list,      # List of ['class', ['token1','token2','token1']]
        train_class_dict_list           # List of ['class', {'token1':2, 'token2':1}]
    ] = __funcs__.extract_data_from_json(input_file_path, tokenizer)

    # 1.3 Make vocabulary.
    print("Making Vocab...")
    train_vocab = __funcs__.make_vocab(train_class_dict_list)   # using list of ['class', {'token1':2, 'token2':1}]

    ''' ---------------- 2. Data Processing ------------------- '''
    print("Summarizing Data ...")
    # 2.1 Initialize token freqs matrix. |vocab|x5 matrix.
    # Row: vocab; Col: classes.
    tf_matrix = pd.DataFrame(np.zeros((len(train_vocab), 6), dtype=int), index=train_vocab)

    # 2.2 Insert values into the matrix.
    for index, row in enumerate(train_class_dict_list):      # For all words in the vocabulary,
        cur_class = row[0]              # class name
        cur_token_dict = row[1]         # e.g. {"apple": 1, "banana":2, ...}
        for key, value in cur_token_dict.items():
            tf_matrix.loc[key, __funcs__.class_map[cur_class]] += value     # Accumulate word freqs for each class
            tf_matrix.loc[key, 5] += value                                  # Also, record in the last row.

    # 2.3 Post process the token frequency matrix.
    tf_matrix = tf_matrix.sort_values(by=5, ascending=False)    # Sort matrix using last row (sum).
    tf_matrix = tf_matrix.iloc[:, :-1]                          # Drop the last row since matrix is sorted.

    ''' ---------------- 3. Visualize Data ------------------- '''
    # 3.1 Print class document frequencies.
    class_freq_str = ""
    for num in train_class_doc_freqs:
        class_freq_str += str(num) + " "
    class_freq_str += "\n"

    # 3.2 Token Frequencies
    token_data_arr = []                         # Rows of the tf-matrix.
    for index, row in tf_matrix.iterrows():
        cur_data_arr = str(index) + " "         # Current word (index of matrix)
        for value in row.values:
            cur_data_arr += str(value) + " "    # The frequencies of this word in each class
        cur_data_arr += "\n"
        token_data_arr.append(cur_data_arr)

    # 3.3 Write files
    if __funcs__.settings['WRITE_FILES'] is False:
        return tf_matrix
    with open(output_file_path, "w") as o_file:
        o_file.write(class_freq_str)
        if __funcs__.settings['PRINT_PROCESS']:
            print(class_freq_str)

        for row in token_data_arr:
            o_file.write(row)
            if __funcs__.settings['PRINT_PROCESS']:
                print(row)
        o_file.close()

    return tf_matrix


#count_word(input_file, output_file)
