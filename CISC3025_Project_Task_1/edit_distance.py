#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# --------------------------------------------------
# Description:  A starter code
# --------------------------------------------------
# Author: Wang-SongSheng <wang.songsheng@connect.um.edu.mo>
# Editor: Huang Yanzhen, DC126732
# Created Date : March 4th 2021, 12:00:00
# --------------------------------------------------

import argparse
import re

# Settings for visualizing & testing algorithms.
# Please set all values to false before running.
custom_settings = {
    "TEST_MODE": False,                  # Run custom_test() func instead of main()
    "PRINT_TABLE": False,                # Print value & operation table.
    "PRINT_TRACK": False,                # Print the backtracked operation array.
    "PRINT_ALIGNMENT_ARRAY": False       # Print the alignment array.
}

# ANSI Colors: For better distinguishablility in console.
color = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "default": "\033[0m"
}

# Pointer Operations Codes
edit = [    # Index -> Code
    '-',    # 0. Match
    'i',    # 1. Insertion
    'd',    # 2. Deletion
    's',    # 3. Substitution
    ' '     # 4. Initialization value, i.e. "Unknown"
]

edit_code = {   # Code -> Index
    'mch': 0,
    'ins': 1,
    'del': 2,
    'sub': 3
}

class Table:
    """
    This class is defined to reduce the extra work of thinking about the data structure of 2D arrays.
    """

    def __init__(self, x_len, y_len, default_val = 0, title = None, word_strings = None, alt_arr = None):
        """
        Constructor for class Table.
        :param x_len: Width of table.
        :param y_len: Height of table.
        :param default_val: Default value to fill in the table when initialized.
        :param word_strings: The two word strings to be compared, used to form headers.
        :param alt_arr: Alternative description array.
        """
        self.x_len = x_len
        self.y_len = y_len
        self.table = [[default_val for i in range(x_len)] for j in range(y_len)]
        self.title = title
        self.word_strings = word_strings
        self.alt_arr = alt_arr

    def get_size(self):
        """
        Get the size of the table.
        :return: A tuple (x,y) representing the table's size.
        """
        return self.x_len, self.y_len

    def get_table(self):
        """
        Get the 2D array embedded in the Table class.
        :return: The 2D array of table data.
        """
        return self.table

    def __is_valid(self, x, y):
        """
        A guardian method to detect insufficient accessing index.
        :param x: X coordinate index.
        :param y: Y coordinate index.
        :return: Whether the (x,y) coordinate is sufficient.
        """
        return 0 <= x <= self.x_len and 0 <= y <= self.y_len

    def read(self, x, y):
        """
        Read a content of table.
        :param x: X coordinate of the data.
        :param y: Y coordinate of the data.
        :return: Data that stores in position (x,y).
        """
        if not self.__is_valid(x, y):
            raise Exception("Can't read: Index out of range.")

        return self.table[y][x]

    def get_last(self):
        """
        Get the last item of the table.
        :return: Last value of the table.
        """
        return self.table[-1][-1]

    def __get_alt(self, code):
        """
        Get an alternative expression.
        :param code:
        :return:
        """
        if self.alt_arr is None:
            return code
        return self.alt_arr[code]

    def __get_input_type(self):
        """
        Get the type of the input.
        :return: Data type of the input.
        """
        if self.word_strings is None:
            return None
        sample = self.word_strings[0]
        return type(sample)

    def write(self, x, y, val):
        """
        Write a value into the table.
        :param x: X coordinate of the data.
        :param y: Y coordinate of the data.
        :param val: The intended value to write into the table.
        """
        if not self.__is_valid(x, y):
            raise Exception("Can't write: Index out of range.")

        # The accessing order needs to be reversed due to the data structure of 2D arrays.
        self.table[y][x] = val

    def fill(self, coord_1, coord_2, val):
        """
        Fill a value into an area of the table.
        :param coord_1: Starting coordinate (x_1, y_1).
        :param coord_2: Ending coordinate (x_2, y_2).
        :param val: The value to be filled in.
        """
        [x_1,y_1] = coord_1
        [x_2,y_2] = coord_2
        for x in range(x_1,x_2 + 1):
            for y in range(y_1,y_2 + 1):
                self.write(x,y,val)

    def levenshtein_init(self, is_val):
        """
        Initialize the table using the init rules of Levenstein distance.
        :param is_val: True for value table or False for pointer table.
        """
        if is_val:
            # Initialize first row of value table.
            for i in range(1, self.x_len):
                self.write(i, 0, i)

            # Initialize first column of value table.
            for i in range(1, self.y_len):
                self.write(0, i, i)
        else:
            # Initialize the first row of pointer table.
            self.write(0, 0, edit_code['mch'])
            self.fill((1, 0), (self.x_len - 1, 0), edit_code['ins'])

            # Initialize the first column of pointer table.
            self.fill((0, 1), (0, self.y_len - 1), edit_code['del'])
        return

    # This function is defined to examine the process of dynamic programming by printing out the table.
    def print_table(self):
        """
        Print the table. Runs in O(n*m) time.
        """
        """ Guardian method, only support print table for char."""
        if not self.__get_input_type() == str:

            return

        ''' Ensure tidiness of table.'''
        # Initialize each column's max width.
        max_widths = [0] * self.x_len

        # Calculate max width for each column.
        for j in range(self.y_len):
            for i in range(self.x_len):
                cell = str(self.read(i, j))
                max_widths[i] = max(max_widths[i], len(cell))


        ''' Print the table.'''
        # Table Title
        if self.title is not None:
            print(self.title + ": ")

        # Pre-process input words.
        if self.word_strings is not None:
            row_word = "#" + self.word_strings[0]
            col_word = "#" + self.word_strings[1]

            # Print first row of input word 1.
            print(" ", end = "  ")          # Table corner

            for i in range(self.x_len):
                print(f"{row_word[i]:<{max_widths[i]}}", end="  ")
            print("\n")

        # Print the table row by row.
        for j in range(self.y_len):
            print(col_word[j], end = "  ")         # Print first column of input word 2.
            for i in range(self.x_len):            # Print rest of the columns.
                code = self.read(i, j)
                print(f"{self.__get_alt(code):<{max_widths[i]}}", end="  ")
            print("\n")

class Node:
    """
    A single-sided tree is required to simplify the alignment process.
    This class represents a node of a tree.
    """
    def __init__(self, val, left = None, mid = None, right = None):
        """
        Constructor.
        :param val: Character value of this node.
        :param left:    Left child.
        :param mid:   Right child.
        """
        self.val = val
        self.left = left
        self.mid = mid
        self.right = right

    def get_val(self):
        """
        Retrieve the value of this node.
        :return: String value.
        """
        return self.val

    def get_left(self):
        """
        Retrieve the left child node.
        :return: Left child node.
        """
        return self.left

    def get_mid(self):
        """
        Retrieve the right child node.
        :return: Right child node.
        """
        return self.mid

    def get_right(self):
        """
        Retrieve the right child node.
        :return: Right child node.
        """
        return self.right

    def set_val(self,val):
        """
        Set the value of this node. Value must be string.
        :param val: Intended value to be set.
        """
        if type(val) != str:
            raise Exception("Value must be a string.")
        self.val = val

    def set_left(self, val):
        """
        Set the value of left child node. Value must be string.
        :param val: Intended value to be set.
        """
        if type(val) != str:
            raise Exception("Child of a node must be a node.")

        if self.left is None:
            self.left = Node([val])
        else:
            arr = self.left.get_val()
            arr.append(val)


    def set_mid(self, val):
        """
        Set the value of right child node. Value must be string.
        :param val: Intended value to be set.
        """
        if type(val) != str:
            raise Exception("Child of a node must be a node.")
        self.mid = Node(val)

    def set_right(self, val):
        """
        Set the value of right child node. Value must be string.
        :param val: Intended value to be set.
        """
        if type(val) != str:
            raise Exception("Child of a node must be a node.")
        self.right = Node(val)

def word_edit_distance(x, y):
    """
    Implements a dynamic programming algorithm to calculate the minimum edit distance & alignment between two words.
    Step 1. Create a table with starting blanks using x and y.
    Step 2. Initialize the table using the init rules of Levenstein distance.
    Step 3. Gradually fill in the table using the rules of Levenstein distance.
    Step 4. Align the words according to the track.

    :param x: Template string, to which operation string is compared with.
    :param y: Operand string, on which operation is performed.
    :return: Minimum edit distance (int) and the alignment 2-D array. Example:
        alignment = [
                    ['a','a','-','a'],
                    ['a','a','b','-']
                ].
    """

    """ Step 1. Create a table with starting blanks using x and y. """
    # First get the length of two strings, and convert to table size.
    # Extra length is required to fill a blank symbol (#).
    table_len_x = len(x) + 1
    table_len_y = len(y) + 1

    # Initialize Value Table
    # An empty table of size (len_x + 1) * (len_y + 1) that is filled with 0.
    val_table = Table(
        table_len_x,
        table_len_y,
        default_val=0,                  # Fill 0 into the table.
        title = "Value Table",
        word_strings = [x,y]
    )

    # Initialize Pointer Table.
    # An empty table of size (len_x + 1) * (len_y + 1) that is filled with "0", meaning not known.
    ptr_table = Table(
        table_len_x,
        table_len_y,
        default_val=-1,                 # Fill "Unknown" into table.
        title = "Operation Table",
        word_strings = [x,y],
        alt_arr = edit
    )

    """ Step 2. Initialize the table using the init rules of Levenstein distance. """

    val_table.levenshtein_init(is_val=True)
    ptr_table.levenshtein_init(is_val=False)

    """ Step 3. Gradually fill in the table using the rules of Levenstein distance. """
    for j in range(1, table_len_y):
        for i in range(1, table_len_x):

            # For a specific cell: Three possible costs.
            sub_cost = val_table.read(i-1,j-1) + (2 if x[i-1] != y[j-1] else 0)         # Substitution
            ins_cost = val_table.read(i-1,j) + 1                                        # Insertion
            del_cost = val_table.read(i,j-1) + 1                                        # Deletion

            # Compare the three costs and decide what to write into two tables.
            costs = [ins_cost, del_cost, sub_cost]
            op_cost = min(costs)        # Operation cost
            op = " "                    # Operation code: sub, ins, del

            # Prioritized selection: When all equal, priority sub > ins > del.
            if sub_cost == op_cost:
                op = edit_code['sub'] if x[i-1] != y[j-1] else edit_code['mch']
            elif ins_cost == op_cost:
                op = edit_code['ins']
            elif del_cost == op_cost:
                op = edit_code['del']

            # Perform write to the two tables.
            val_table.write(i, j, op_cost)
            ptr_table.write(i, j, op)

    # If required, print the tables in console.
    if custom_settings['PRINT_TABLE']:
        val_table.print_table()
        ptr_table.print_table()

    # Last num is the edit distance.
    edit_distance = val_table.get_last()

    """ Step 4. Align the words according to the track."""
    ''' 4-1. Pre-process the two words.'''
    # 4-1.1. Convert two words into two arrays.
    x_arr = [ch for ch in x]
    y_arr = [ch for ch in y]

    # 4-1.2 Align strings with different length into same length.
    # Note: I found out this is not necessary....
    [x_arr, y_arr] = align_length(x_arr, y_arr)

    # 4-1.3. Insert array element into a single-sided tree.
    x_init_node = Node("#")
    y_init_node = Node("#")

    [x_init_ptr, y_init_ptr] = [x_init_node, y_init_node] # Initialize two lag pointers.

    for ch in x_arr:        # Form the tree of x.
        x_init_ptr.set_mid(ch)
        x_init_ptr = x_init_ptr.get_mid()

    for ch in y_arr:        # Form the tree of y.
        y_init_ptr.set_mid(ch)
        y_init_ptr = y_init_ptr.get_mid()

    ''' 4-2. Track the operation path from the operation table.'''
    # 4-2.1. Track the path from the table (in reversed order).
    op_track = track_ptr_table(ptr_table)
    op_track = op_track[::-1]       # Reverse the operation track

    if custom_settings['PRINT_TRACK']:
        print(color['yellow'] + "Operation Track:" + color['default'])
        print(op_track)

    ''' 
    4-3. Traverse the tree, add hyphen mark to the other side 
    of the tree when required (ins, del).
    '''
    # 4-3.1. Initialize array pointer and track pointer.
    track_ptr = 0
    x_node_ptr = x_init_node  # First letter of x array
    y_node_ptr = y_init_node  # First letter of y array

    # 4-3.2. Traversing the tree, adding hyphens.
    while x_node_ptr is not None and y_node_ptr is not None:
        # Stop criteria: One of them reaches end (in this case words have different length).
        if op_track[track_ptr] == edit_code['ins']:
            # Insertion.
            # Operand string (x): Insert a "-" before cur letter.
            y_node_ptr.set_left("-")
            x_node_ptr = x_node_ptr.get_mid()
        elif op_track[track_ptr] == edit_code['del']:
            # Deletion.
            # Template string (y): Insert a "-" after current letter.
            x_node_ptr.set_left("-")
            y_node_ptr = y_node_ptr.get_mid()
        elif op_track[track_ptr] == edit_code['sub'] or op_track[track_ptr] == edit_code['mch']:
            # Proceed
            x_node_ptr = x_node_ptr.get_mid()
            y_node_ptr = y_node_ptr.get_mid()

        track_ptr += 1

    # 4-3.3. Pre-order the tree again to retrieve the processed array.
    x_arr = traverse(x_init_node)
    y_arr = traverse(y_init_node)

    ''' 4.4. Post Processing to Restore array.'''
    # 4.4.1. Remove the blank buffer.
    x_arr = [ch for ch in x_arr if ch != "#"]
    y_arr = [ch for ch in y_arr if ch != "#"]

    # 4.4.2. Reverse the array again to get the alignment.
    alignment = [x_arr,y_arr]

    # If required, print the alignment array.
    if custom_settings['PRINT_ALIGNMENT_ARRAY']:
        print(color['yellow'] + "Alignment:" + color['default'])
        print(alignment[0])
        print(alignment[1])

    return edit_distance, alignment

def track_ptr_table(ptr_table):
    """
    This function back-tracks an operation table and retrieves
    a reversed array of the sequence of operations.
    :param ptr_table: Pointer table to be tracked.
    :return: An array of reversed operation sequence (in operation code).
    """

    # Get the size of the table.
    [table_len_x, table_len_y] = ptr_table.get_size()

    # Define the current position.
    [cur_x, cur_y] = [table_len_x - 1, table_len_y - 1]

    # Initialize the reversed operation track array.
    op_track = []

    while cur_x >= 0 and cur_y >= 0:
        # Stop criteria: One of them reaches end (in this case words have different length).
        # Append current operand into array (include 'match').
        cur_op = ptr_table.read(cur_x, cur_y)
        op_track.append(cur_op)

        # Change the next position according to current op code.
        if cur_op == edit_code['sub'] or cur_op == edit_code['mch']:
            # Substitution or match, move diagnally.
            [cur_x, cur_y] = [cur_x - 1, cur_y - 1]

        elif cur_op == edit_code['ins']:
            # Insertion, ptr of template string change, ptr of operation string don't.
            cur_x -= 1

        elif cur_op == edit_code['del']:
            # Deletion, ptr of template string don't change, ptr of operation string change.
            cur_y -= 1

        else:
            raise Exception('ERROR: Unknown operation code.')

    return op_track

def traverse(root, result = None):
    """
    This function reads a root node of a single-sided tree, and pre-order
    traverse the tree, which may contain hyphens, indicating there is an insertion
    or deletion. The traversal embraces the hyphen into the sequence, restoring
    the reversed aligned array.
    :param root: Root node of a word letter array, most likely defined specially as "#".
    :param result: Intermediate array for recursion, which shouldn't be inputted externally.
    :return: Reversed aligned array, e.g.['Y','P','P','-','A',''H].
    """
    if result is None:
        result = []
    if root is not None:
        # Left node
        traverse(root.get_left(),result)
        # Visit root
        if type(root.get_val()) == list:
            result.extend(root.get_val())
        else:
            result.append(root.val)
        # Right node
        traverse(root.get_right(), result)
        # Next node
        traverse(root.get_mid(), result)
    return result

def align_length(x_arr,y_arr):
    """
    Align strings of different length into same length.
    :param x_arr: First string.
    :param y_arr: Second string.
    :return: Aligned strings.
    """
    if len(x_arr) == len(y_arr):
        return x_arr, y_arr

    # Find whichever array that's shorter
    short_arr = x_arr if len(x_arr) < len(y_arr) else y_arr
    long_arr = x_arr if short_arr == y_arr else y_arr
    is_x_shorter = short_arr == x_arr

    # Number of len(long_arr)-len(short_arr) is appended to short array.
    num_of_blanks = len(long_arr) - len(short_arr)
    short_arr.extend(["#"] * num_of_blanks)

    # Restoring
    x_arr = short_arr if is_x_shorter else long_arr
    y_arr = short_arr if x_arr == long_arr else long_arr

    return x_arr, y_arr

def sentence_edit_distance(x, y):
    """
    Implement the dynamic programming algorithm to calculate the edit distance
    and alignment between two sentences.
    :param x: Template token list, to which operand list is compared.
    :param y: Operand list, on which operation is performed.
    :return: Edit distance  & alignment array. An example:
        alignment = [
                    ['Today','is','a','good','day'],
                    ['Today','-','a','good','day']
                ].
    """
    """ Step 1. Pre-process sentences into token arrays."""
    # This is omitted. It is done outside the function.
    #[x_arr,y_arr] = [sentence_preprocess(x),sentence_preprocess(y)]

    """ Step 2. Apply this array to match."""
    edit_distance,alignment = word_edit_distance(x,y)

    return edit_distance,alignment

def sentence_preprocess(sentence):
    """
        Temporarily preprocess the sentence string input from the command line.

        :param sentence: A string sentence.
        :return: The tokenized sentence (a list, each item corresponds to a word or a punctuation of the sentence)
    """

    # Define the splitting delimiters using regular expression.
    rule = r'[\s\~\`\!\@\#\$\%\^\&\*\(\)\-\_\+\=\{\}\[\]\;\:\'\"\,\<\.\>\/\?\\|]+'
    re.compile(rule)

    # Store distinct tokens into array.
    # This may contain empty member '' (empty string).
    tokens_ = []
    # Since we consider it case-sensitive, no need to convert to lowercase here.
    tokens_ = tokens_ + re.split(rule, sentence)

    # Remove the potential empty member ''
    tokens = []
    for term in tokens_:
        if term != '':
            tokens.append(term)

    return tokens

def output_alignment(alignment):
    #output the alignment in the format required
    if len(alignment[0]) != len(alignment[1]):
        print('ERROR: WRONG ALIGNMENT FORMAT')
        input()
        exit(0)
    print('An possible alignment is:')
    merged_matrix = alignment[0] + alignment[1]
    max_len = 0
    for item in merged_matrix:
        if len(item) > max_len:
            max_len = len(item)
    for i in range(len(alignment[0])):
        print (alignment[0][i].rjust(max_len)+' ',end=''),
    print('')
    for i in range(len(alignment[0])):
        print (('|').rjust(max_len) + ' ',end=''),
    print('')
    for i in range(len(alignment[1])):
        print (alignment[1][i].rjust(max_len)+' ',end='')
    print('')
    return

def batch_word(input_file, output_file=None):
    """
    Read an input file with H/R started lines and outputs the processed min edit distance.
    :param input_file: File path of input file.
    :param output_file: File path of output file.
    """
   # Open files, store lines into array.
    with open(input_file, "r") as file:
        data = file.readlines()

    # Define a rule to split the line into code and words.
    rule= r'[\s]+'
    re.compile(rule)

    # Start to process.
    cur_anchor = ""         # Code-R words
    code_and_words = []     # Stored instances of code, words and edit dist.
    for token in data:
        code_and_word = re.split(rule,token)
        # I used to remove the empty member, but it seems unnecessary.
        # code_and_word = [ch for ch in code_and_word if ch != ""]
        if code_and_word[0]=="R":
            # Meeting an R, change the anchor word to this.
            cur_anchor = code_and_word[1]
            code_and_words.append(code_and_word)
        elif code_and_word[0]=="H":
            # Meeting an H, compare this with anchor word.
            [edit_distance,_] = word_edit_distance(cur_anchor,code_and_word[1])
            # Store edit distance at last slot.
            code_and_word[2] = str(edit_distance)
            code_and_words.append(code_and_word)
        else:
            # Shouldn't meet something other than R or H.
            raise Exception("Invalid header code!")

    # Initialize output
    output = ""
    for code_and_word in code_and_words:
        item = code_and_word[0] + " " + code_and_word[1] + " " + code_and_word[2] + "\n"
        output = output + item
    print(output)

    # Write output to external file.
    if output_file is not None:
        with open(output_file,"w") as o_file:
            o_file.write(output)

def batch_sentence(input_file,output_file=None):
    """
    Read an input file with H/R started lines and outputs the processed min edit distance.
    :param input_file: File path of input file.
    :param output_file: File path of output file.
    """
    with open(input_file, "r") as file:
        data = file.readlines()

    # Define a rule to split the line into code and sentences.
    rule= r'(?<=[H|R])'
    re.compile(rule)

    # Start to process.
    cur_anchor = ""         # Code-R words
    code_and_sentences = []     # Stored instances of code, words and edit dist.
    for token in data:
        code_and_sentence = re.split(rule,token)
        # Remove empty members.
        code_and_sentence = [ch for ch in code_and_sentence if ch != ""]
        if code_and_sentence[0] == "R":
            # Meeting an R, change the anchor sentenc to this.
            code_and_sentence.append("")
            cur_anchor = code_and_sentence[1]
            code_and_sentences.append(code_and_sentence)
        elif code_and_sentence[0] == "H":
            # Meeting an H, compare this with anchor sentence.
            [edit_distance, _] = sentence_edit_distance(cur_anchor, code_and_sentence[1])
            # Store edit distance at last slot.
            code_and_sentence.append(str(edit_distance))
            code_and_sentences.append(code_and_sentence)
        else:
            # Shouldn't meet something other than R or H.
            raise Exception("Invalid header code!")
        print(code_and_sentence)

    # Initialize output
    output = ""
    for code_and_sentence in code_and_sentences:
        sentence_content = code_and_sentence[1]
        sentence_content.replace("\n"," ")
        item = code_and_sentence[0] + " " + sentence_content + " " + code_and_sentence[2] + " \n"
        output = output + item

    # Write output to external file.
    if output_file is not None:
        with open(output_file,"w") as o_file:
            o_file.write(output)

    print(output)

    return


def main():
    """
    Main Function.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--word',type=str,nargs=2,help='word comparson')
    parser.add_argument('-s','--sentence',type=str,nargs=2,help='sentence comparison')
    parser.add_argument('-bw','--batch_word',type=str,nargs=2,help='batch word comparison,input the filename')
    parser.add_argument('-bs','--batch_sentence',type=str,nargs=2,help='batch word comparison,input the filename')

    opt=parser.parse_args()

    if(opt.word):
        edit_distance,alignment = word_edit_distance(opt.word[0],opt.word[1])
        print('The cost is: '+str(edit_distance))
        output_alignment(alignment)
    elif(opt.sentence):
        edit_distance,alignment = sentence_edit_distance(sentence_preprocess(opt.sentence[0]),sentence_preprocess(opt.sentence[1]))
        print('The cost is: '+str(edit_distance))
        output_alignment(alignment)
    elif(opt.batch_word):
        batch_word(opt.batch_word[0],opt.batch_word[1])
    elif(opt.batch_sentence):
        batch_sentence(opt.batch_sentence[0],opt.batch_sentence[1])

def custom_test():

    custom_test_settings = {
        "TEST_WORD": True,
        "TEST_SENTENCE": False,
        "TEST_WORD_CORPUS":False,
        "TEST_SENTENCE_CORPUS":False,
    }

    test_subjects_word=[
        #("EXECUTION", "INTENTION"),     # Course example
        ("ALIGN","ALIGNMENT"),          # Wierd, seems to match further letters.
        #("LAND","LANDLORDS"),
        #("LAND","LANDLORNS"),
        #("F","FLOW"),
        #("","DYNAMIC"),
        #("ALIGN","ALIGNMENT"),
        #("AGGCTATCAC","TAGCTGTCAC"),    # Alternative ins and del
        #("HAPPY","HAPPY"),              # Exact same
        #("EXTENSION", "INTENTION"),     # Different but no ins or del
        #("AB","A"),                     # Double letter
        #("A","B"),                      # Single letter
        #("","A"),                       # Empty string & single letter
        #("","")                         # Empty string
        #("AGGCTATCACCTGACCTCCAGGCCGATGCCC","TAGCTATCACGACCGCGGTCGATTTGCCCGAC")
    ]

    test_subjects_sentence = [
        (
            "I love natural language processing.",
            "I really like natural language processing course."
         ),
        (
            "I love you.",
            "I love you."
        ),
        (
            "I like the cake.",
            "The cake is a lie."
        ),
        (
            "I am a fool.",
            "I am the fool."
        ),
        (
            "I am a fool",
            "Am I the fool?"
        ),
        (
            "AAAAAA AAAA AA A",
            "SFS FFF AA f"
        )
    ]

    if custom_test_settings["TEST_WORD"]:
        for index, test_subject in enumerate(test_subjects_word):
            print(color['green'] + "\nTest " + str(index+1) + ": " + color['default'], end = " ")
            print(test_subject[0] + " & " + test_subject[1])
            [min_edit_dist, alignment] = word_edit_distance(test_subject[0],test_subject[1])
            print(color['yellow'] + "Minimal Edit Distance: " + color['default'] + str(min_edit_dist))

    if custom_test_settings['TEST_SENTENCE']:
        for index, test_subject in enumerate(test_subjects_sentence):
            print(color['green'] + "\nTest " + str(index+1) + ": " + color['default'], end = " ")
            print(test_subject[0] + " & " + test_subject[1])
            [min_edit_dist, alignment] = sentence_edit_distance(test_subjects_sentence[0],test_subjects_sentence[1])
            print(color['yellow'] + "Minimal Edit Distance: " + color['default'] + str(min_edit_dist))

    if custom_test_settings['TEST_WORD_CORPUS']:
        batch_word("./InputFiles/word_corpus.txt", "./OutputFiles/word_edit_distance.txt")
    if custom_test_settings['TEST_SENTENCE_CORPUS']:
        batch_sentence("./InputFiles/sentence_corpus.txt","./OutputFiles/sentence_edit_distance.txt")

if __name__ == '__main__':
    import os
    # Determined by the TEST_MODE settings at custom_settings.
    main() if custom_settings["TEST_MODE"] == False else custom_test()
