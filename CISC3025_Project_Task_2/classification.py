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
    "WRITE_DATA": True
}

input_file = './data/test.json'
output_file = './output/classification_result.txt'
temp_output_file = "./temp_output/classification_compare.txt"


def classification(input_file_path, output_file_path, temp_output_file_path):
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

    # 2.3 For each ['word',['prob1','prob2','prob3','prob4','prob5']]:
    # Convert string to float. Get ['w',[ -log[P(w|c1)], -log[P(w|c2)], -log[P(w|c3)], -log[P(w|c4)], -log[P(w|c5)]]].
    log_neg_word_probs_dict = {}
    for instance in _log_neg_word_probs_arr:
        # 4.3.1 Basic data.
        cur_word = instance[0]                   # Current Word
        _cur_log_neg_word_probs = instance[1]    # Current str(-log[P(w|c)]) for all c in C of this word.

        # 4.3.2 Convert float to string.
        cur_log_neg_word_probs = [float(num) for num in _cur_log_neg_word_probs]     # -log[P(w|c)] for all c
        log_neg_word_probs_dict[cur_word] = cur_log_neg_word_probs

    # ------------- 3. Calculate Naive Bayes Prob -------------#
    # 3.1 Retrieve test data.
    [
        test_class_freq,            # [freq1, freq2, freq3, freq4, freq5]
        test_word_sentence_list,    # ['class', ['word1', 'word2', 'word1',...]]
        test_word_dict_list         # ['class', {'word1':2, 'word2':1,...}]
    ] = __funcs__.extract_data_from_json(input_file_path, tokenizer, include_id=True)

    # 3.2 Retrieve training data.
    [
        train_class_freq,            # [freq1, freq2, freq3, freq4, freq5]
        train_word_sentence_list,    # ['class', ['word1', 'word2', 'word1',...]]
        train_word_dict_list         # ['class', {'word1':2, 'word2':1,...}]
    ] = __funcs__.extract_data_from_json("./data/train.json", tokenizer)

    # 3.2 Create vocabulary for training data.
    vocab = set()
    for instance in train_word_dict_list:
        for word in instance[1]:
            vocab.add(word)
    vocab = list(vocab)

    # 3.3 Classify
    class_pred_tuples = []
    for instance in test_word_sentence_list:
        # For each  ['class', ['word1', 'word2', 'word1',...]]
        test_cur_class = instance[0]        # Current Class
        test_cur_word_arr = instance[1]     # Current Sentence ['word1', 'word2', 'word1',...]
        test_cur_file_id = instance[2]      # Current File ID

        # 3.3.1 Form a probability matrix for each sentence.
        # - Column Index: Word tokens in this sentence.
        # - Row Index: Class.
        # - Content: -log[P(w|c)], where w, c are the row & column indexes respectively.
        cur_sentence_word_probs_list = []   # [[prob11, prob12, .., prob15],[prob21, prob22, .., prob25]..]
        for word in test_cur_word_arr:
            # For each word in ['word1', 'word2', 'word1',...]
            if word in log_neg_word_probs_dict:
                # If present in training vocab, get the list of joined probs
                cur_log_neg_word_probs = log_neg_word_probs_dict[word]  # Current

            else:
                # If not, set the list to empty_prob_val = -log(1 / (word freqs for class + vocab size))
                cur_log_neg_word_probs = [
                    -math.log(1 / (word_freqs + (len(vocab)+1)))
                    for word_freqs in train_class_freq
                ]
            cur_sentence_word_probs_list.append(cur_log_neg_word_probs)

        # 3.3.2 Calculate P(sentence|c) for all class in negative log space.
        cur_sentence_joined_probs = [0, 0, 0, 0, 0]
        for log_neg_probs in cur_sentence_word_probs_list:
            for index, log_neg_prob in enumerate(log_neg_probs):
                cur_sentence_joined_probs[index] += log_neg_prob

        # 3.3.3 Multiply P(sentence|c) with P(c) for all class in negative log space.
        cur_sentence_joined_probs = [
            num + log_neg_doc_probs_for_each_class[index]
            for index, num in enumerate(cur_sentence_joined_probs)
        ]

        # 3.3.4 Record data.
        # Since negative log space is used, the smaller the num the higher the prob.
        prediction = __funcs__.classes[np.argmin(cur_sentence_joined_probs)]
        cur_class_pred_tuple = [test_cur_file_id, prediction, test_cur_class]
        class_pred_tuples.append(cur_class_pred_tuple)
        print(cur_class_pred_tuple[0], " ", cur_class_pred_tuple[1], " ", cur_class_pred_tuple[2])

    # ------------ 4. Write Data ------------ #
    with open(output_file_path, 'w') as o_file:
        for instance in class_pred_tuples:
            instance_str = instance[0] + " " + instance[1] + "\n"
            o_file.write(instance_str)
        o_file.close()

    with open(temp_output_file_path, 'w') as to_file:
        to_file.write("\n")
        for instance in class_pred_tuples:
            instance_str = instance[0] + " " + instance[1] + " " + instance[2] + "\n"
            to_file.write(instance_str)
        to_file.close()


#classification(input_file, output_file, temp_output_file_path)
