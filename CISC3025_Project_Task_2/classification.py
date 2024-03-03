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
    # Tokenizer: Split sentence using space characters.
    tokenizer = nltk.tokenize.RegexpTokenizer(r'[\s]+', gaps=True)

    # ------------ 2. Process Data -------------#
    [
        test_class_freq,
        test_word_sentence_list,
        test_word_dict_list
    ] = __funcs__.extract_data_from_json(input_file_path, tokenizer)

    # ---------- 3. Make Vocabulary ------------#
    vocab = set()
    for instance in test_word_dict_list:
        for word in instance[1]:
            vocab.add(word)
    vocab = list(vocab)

    # -------- 4. Prepare Joined Probs ---------#
    # 4.1 Retrieve logged and negatived values of all probs.
    # - First param: -log[P(c)] for all c in C. C is the set of all classes.
    # - Second param: List of word-prob pairs. In each pair, let word be w, then probs are -log[P(w|c)] for all c.
    # - Sample row of second param: ['word', ['val1', 'val2', 'val3', 'val4', 'val5']]
    [
        log_neg_doc_probs_for_each_class,       # -log[P(c)]
        log_neg_word_probs_arr                  # -log[P(w|c)] for all w in V.
    ] = __funcs__.extract_data_from_txt(
        "./temp_output/word_probability_log.txt",
        tokenizer,
        include_first_line=True,
        convert_to_int=False)

    # 4.2 For -log[P(c)] for all c in C, convert string to float.
    log_neg_doc_probs_for_each_class = [float(num) for num in log_neg_doc_probs_for_each_class]

    # 4.3 For each word-probs pairs, convert string to float. And calculate joined probs.
    train_data_vocab = []
    log_neg_word_joined_probs = []
    for instance in log_neg_word_probs_arr:
        # 4.3.1 Basic data.
        cur_word = instance[0]
        train_data_vocab.append(cur_word)

        log_neg_cur_word_probs = instance[1]

        # 4.3.2 Convert float to string.
        log_neg_cur_word_probs = [float(num) for num in log_neg_cur_word_probs]     # -log[P(w|c)] for all c

        # 4.3.3 Compute joined probs in log space: -log[P(w|c)*P(c)] = -log[P(w|c)] + -log[P(c)]
        log_neg_cur_word_join_probs = [
            num + log_neg_doc_probs_for_each_class[_class]
            for _class, num in enumerate(log_neg_cur_word_probs)]

        # 4.3.4 Restore joined probability, get P(w|c)*P(c) for all words.
        # cur_word_join_probs = [math.exp(-num) for num in log_neg_cur_word_join_probs]
        log_neg_word_joined_probs.append([cur_word, log_neg_cur_word_join_probs])

    # for instance in log_neg_word_joined_probs:
    #     print(instance[0], end=" ")
    #     print(instance[1])

    # ------------- 5. Calculate Naive Bayes Prob -------------#
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