'''
CISC3025-Project-Task-02 Requirement 1: Count words
'''

import os
import json
import re
import nltk
import pandas as pd
import numpy as np
from collections import Counter

input_file = './data/train.json'
output_file = './output/word_count.txt'

custom_settings = {
    "WRITE_DATA": False
}


def count_word(input_file_path, output_file_path):
    # ---------- 1. Preparation ------------#
    print("Opening File...")
    # 1.1 Open file, read json data.
    with open(input_file_path, 'r') as f:
        train_data = json.load(f)

    print(train_data)

    # 1.2 Map class name to index number.
    class_map = {
        'crude': 0,
        'grain': 1,
        'money-fx': 2,
        'acq': 3,
        'earn': 4
    }
    classes = ['crude', 'grain', 'money-fx', 'acq', 'earn']

    # 1.3 Initialize the frequency vector of classes.
    # Order: crude, grain, money-fx, acq, earn
    class_freq = [0, 0, 0, 0, 0]

    # 1.4 Prepare tokenizer
    # Tokenizer: Split sentence using space characters.
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\s]+', gaps=True)

    # 1.4 Process each instance of test_data. Instance content: [train_id, class, sentence]
    tokenized_test_data = []        # Instance example: [class_name, ['token1', 'token2','token1']]
    tokenized_test_data_freq = []   # Instance example: [class_name, {'token1':2, 'token2':1}]

    # ------------ 2. Process Data -------------#
    print("Processing Data...")
    for instance in train_data:
        # 2.1 Record Class, count doc num for each class.
        cur_class = instance[1]
        class_freq[class_map[cur_class]] += 1

        # 2.2 Tokenize sentence part of each instance of train data.
        cur_sentence = instance[2]

        # 2.2.1 Replace common delimiters by a single space and eliminate some useless characters.
        # Remove double-commas, abbreviation marks, brackets, and continuous hyphens.
        cur_sentence = re.sub(r'\"|\.\.+|\(|\)|\s--+\s|(?<=[A-Za-z])/|&[a-z]+;|>', ' ', cur_sentence)
        # Remove period, comma, question mark and exclamation mark followed by a  space as common delimiters.
        # Leave abbreviations alone, like U.S. or U.K.
        cur_sentence = re.sub(r'(?<![A-Z])([.,?!"]\s+)', ' ', cur_sentence)

        # 2.2.2 Tokenize current sentence.
        _cur_token_array = tokenizer.tokenize(cur_sentence)
        cur_token_array = []

        # 2.2.3 Post processing:
        # Remove remaining cases where there's a punctuation mark at the end of a word.
        for word in _cur_token_array:
            word = word.rstrip(',?!"-')
            cur_token_array.append(word) if word != "-" or "" else None

        # 2.3 Summarize Data
        # 2.3.1 Array Part
        t_test_data_instance = [cur_class, cur_token_array]
        tokenized_test_data.append(t_test_data_instance)
        # 2.3.2 Frequency part
        cur_freq_dict = Counter(cur_token_array)
        cur_freq_dict = dict(cur_freq_dict)
        t_test_data_freq_instance = [cur_class, cur_freq_dict]
        tokenized_test_data_freq.append(t_test_data_freq_instance)

    # ----------------3. Make Vocabulary ---------------#
    print("Making Vocab...")
    vocab = set()
    for instance in tokenized_test_data:
        for word in instance[1]:
            vocab.add(word)
    vocab = list(vocab)

    # -------4. Summarize instances for each word.-------#
    print("Summarizing Data ...")
    # 4.1 Initialize term frequency matrix using vocabulary as index.
    # Row num: Vocab size; Col num: Num of class, i.e. 5. Last row: sum of each row.
    tf_matrix = pd.DataFrame(np.zeros((len(vocab), 6), dtype=int), index=vocab)

    # 4.2 For all words in the vocabulary
    for index, row in enumerate(tokenized_test_data_freq):
        cur_class = row[0]
        cur_token_dict = row[1]     # e.g. {"apple": 1, "banana":2, ...}
        for key, value in cur_token_dict.items():
            # Accumulate corresponding value of instance to the correct class in the matrix.
            tf_matrix.loc[key, class_map[cur_class]] += value
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
        return
    with open(output_file_path, "w") as o_file:
        o_file.write(class_freq_str)
        for row in token_data_arr:
            o_file.write(row)
        o_file.close()

    return tf_matrix


count_word(input_file, output_file)