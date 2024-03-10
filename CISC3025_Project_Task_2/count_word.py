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

custom_settings = {
    "WRITE_DATA": True
}


def count_word(input_file_path, output_file_path):
    # ---------- 1. Preparation ------------#
    # Tokenizer: Split sentence using space characters.
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\s]+', gaps=True)

    # ---------- 2. Process Data -----------#
    [
        class_freq,
        train_class_sentence_list,
        train_class_dict_list
    ] = __funcs__.extract_data_from_json(input_file_path, tokenizer)

    # ----------------3. Make Vocabulary ---------------#
    print("Making Vocab...")
    vocab = set()
    for instance in train_class_sentence_list:
        for word in instance[1]:
            vocab.add(word)
    vocab = list(vocab)

    # -------4. Summarize instances for each word.-------#
    print("Summarizing Data ...")
    # 4.1 Initialize term frequency matrix using vocabulary as index.
    # Row num: Vocab size; Col num: Num of class, i.e. 5. Last row: sum of each row.
    tf_matrix = pd.DataFrame(np.zeros((len(vocab), 6), dtype=int), index=vocab)

    # 4.2 For all words in the vocabulary
    for index, row in enumerate(train_class_dict_list):
        cur_class = row[0]
        cur_token_dict = row[1]     # e.g. {"apple": 1, "banana":2, ...}
        for key, value in cur_token_dict.items():
            # Accumulate corresponding value of instance to the correct class in the matrix.
            tf_matrix.loc[key, __funcs__.class_map[cur_class]] += value
            # Also, record this accumulation in the last row.
            tf_matrix.loc[key, 5] += value

    # 4.3 Post process the token frequency matrix.
    # Sort this matrix using the last row.
    tf_matrix = tf_matrix.sort_values(by=5, ascending=False)
    # Since this matrix is sorted, drop the last row.
    tf_matrix = tf_matrix.iloc[:, :-1]

    # --------------5. Print and write data---------------#
    # 5.1 Print class data
    class_freq_str = ""
    for num in class_freq:
        class_freq_str += str(num) + " "
    class_freq_str += "\n"

    # 5.2 Print Term frequency
    token_data_arr = []
    for index, row in tf_matrix.iterrows():
        # Current word
        cur_data_arr = str(index) + " "
        # The frequencies in each class
        for value in row.values:
            cur_data_arr += str(value) + " "
        cur_data_arr += "\n"
        token_data_arr.append(cur_data_arr)

    # 5.3 Write files
    if custom_settings['WRITE_DATA'] is False:
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
