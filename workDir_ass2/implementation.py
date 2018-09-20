import tensorflow as tf
import sys, string, re
import numpy as np

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 150  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
NUM_CLASSES = 2

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

other_punctuation = set({"<br />"})



# Print original and modified reviews (Forms: string // list of words)
# for DEBUG
def print_review(func):
    def wrapper(review):
        print(review)

        processed_review = func(review)

        for word in processed_review:
            print(word)
        a = input("==========\n=Continue=\n==========\n")
        return processed_review
    return wrapper

#@ print_review # Deisplay comment onto review for competingch
def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    # Lower case
    processed_review = review.lower()


    # Strip punctions. Replace them with " " for future splitting
    for punc in other_punctuation:
        processed_review = processed_review.replace(punc, " ")

    # Remove necessary punctuation
    # !!Reserve words containing '-' char, such as "f-ing"
    for punc in string.punctuation.replace('-',''):
        processed_review = processed_review.replace(punc, " ")

    # Split sentence by " "
    processed_review = processed_review.split()

    # Remove stop words
    processed_review = [word for word in processed_review if word not in stop_words]
    return processed_review




"""

l_configs = {
    "single": {
        "learning_rate": 0.001,
        "hidden_units": [128,]
    },
    "double": {
        "learning_rate": 0.001,
        "hidden_units": [100,110]
    }
}

class Config(object):
    default_config = {
        "dtype": tf.float32,
        "learning_rate": 0.1,
        "hidden_units": 64,
        "default_dropout_keep_prob": 0.75,
        "rnn_cell": tf.nn.rnn_cell.LSTMCell,
        "rnn_cell_initializer": tf.random_normal_initializer,
        "rnn_net": tf.nn.dynamic_rnn,
        "optimizer": tf.train.AdamOptimizer,
        "dynamic_seq": False
    }

    def __init__(self, input):
        for p in self.default_config:
            setattr(self, p, input.get(p,self.default_config[p]))




def get_seq_len(m):
    one_or_zero = tf.sign(tf.reduce_max(tf.abs(m), 2))
    length = tf.cast(tf.reduce_sum(one_or_zero, 1), tf.int32)
    return length

def create_cell(c, n, dropout_keep_prob):
    cell = c.rnn_cell(n, initializer=c.rnn_cell_initializer())
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)


def define_graph():
"""
"""
    Implement your model here. You will need to define placeholders, for the inputs and labels,
    Note that the inputs is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
"""
"""
    # Select configurations
    c = Config(l_configs["double"])

    # Define placeholders
    input_data = tf.placeholder(tf.float32,shape=[BATCH_SIZE,MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE],name='input_data')
    labels = tf.placeholder(tf.float32, shape = [BATCH_SIZE,NUM_CLASSES],name = 'labels')
    dropout_keep_prob = tf.placeholder_with_default(c.default_dropout_keep_prob, shape=(), name='dropout_keep_prob')

    # Obtain results
    if len(c.hidden_units) == 1:
        cell = create_cell(c, c.hidden_units[0], dropout_keep_prob)
    else:
        cell = tf.contrib.rnn.MultiRNNCell([create_cell(c,n,dropout_keep_prob) for n in c.hidden_units])

    seq_len = get_seq_len(input_data) if c.dynamic_seq else None
    outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=input_data, sequence_length=seq_len, dtype=c.dtype)

    last_output = outputs[:,-1,:]
    logits = tf.layers.dense(last_output, NUM_CLASSES)

    correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'accuracy')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels), name = 'loss')
    optimizer = c.optimizer(c.learning_rate).minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
"""

l_configs = {
    # validate acc: 0.82 (ite:8000)
    "safe": {
        "dtype": tf.float32,
        "learning_rate": 0.1,
        "hidden_units": [[64,]],
        "default_dropout_keep_prob": 0.75,
        "rnn_cell": tf.contrib.rnn.LSTMCell,
        "rnn_cell_initializer": tf.orthogonal_initializer,
        "rnn_net": tf.nn.dynamic_rnn,
        "optimizer": tf.train.AdamOptimizer,
        "dynamic_seq": False
    },
    # v 0.8463 (2000)
    # v 0.8380 (3000)
    # v 0.810  (4000)
    "trial1": {
        "hidden_units": [[128,128]],
        "extra_dense_activation": tf.nn.sigmoid,
    },
    # v 0.8406 (1600)
    # v 0.819 (2400)
    # v 0.8344 (3400)
    # v 0.8188 (4300)
    "trial2": {
        "hidden_units": [[128,128]],
        "extra_dense_units": [[128,]],
        "extra_dense_activation": tf.nn.sigmoid,
    },
    # v 0.8406 (1500)
    # v 0.8276 (2500)
    # v 0.8198 (4000)
    # v 0.8271 (6000)
    "trial3": {
        "hidden_units": [[64]],
        "extra_dense_units": [[128,]],
        "extra_dense_activation": tf.nn.sigmoid,
        "learning_rate": 0.001,
    },
    # v 0.816 (10000)
    "trial4": {
        "hidden_units": [[64]],
        "extra_dense_units": [[128, ]],
        "extra_dense_activation": tf.nn.relu,
    },
    #
    "trial5": {
        "hidden_units": [[64,]],
        "extra_dense_units": [[128, ]],
        "extra_dense_activation": tf.nn.relu,
    },
    # v 0.81875 (8000)
    "ensemble1": {
        "hidden_units": [[128],[64,64]],
        "extra_dense_units": [[32,]],
        "extra_dense_weights_initializer": tf.random_normal_initializer
    },
    "ensemble2": {
        "hidden_units": [[128,128],[64,100], [100]],
        "extra_dense_units": [[128,], [128,], [64,]],
        "extra_dense_activation": tf.nn.sigmoid,
    }
}

class Config(object):
    default_config = {
        "dtype": tf.float32,
        "hidden_units": [[64,]],
        "dynamic_seq": False,
        "rnn_cell": tf.contrib.rnn.LSTMCell,
        "rnn_cell_initializer": tf.orthogonal_initializer,

        "extra_dense_units": [[]],
        "extra_dense_activation": tf.nn.relu,
        "extra_dense_dropout_keep_prob": 1.0,
        "extra_dense_weights_initializer": tf.truncated_normal_initializer,
        "extra_dense_bias_initializer": tf.zeros_initializer,

        "learning_rate": 0.1,
        "optimizer": tf.train.AdamOptimizer,

    }

    def __init__(self, input):
        for p in self.default_config:
            setattr(self, p, input.get(p,self.default_config[p]))

def get_seq_len(m):
        a = tf.reduce_max(tf.abs(m), 2)
        one_or_zero = tf.sign(a)
        length = tf.cast(tf.reduce_sum(one_or_zero, 1), tf.int32)
        return length

def create_cell(c, n, dropout_keep_prob):
    cell = c.rnn_cell(n, initializer=c.rnn_cell_initializer())
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the inputs and labels,
    Note that the inputs is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    # Select configurations
    c = Config(l_configs["trial4"])

    # Define placeholders
    input_data = tf.placeholder(tf.float32,shape=[BATCH_SIZE,MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE],name='input_data')
    labels = tf.placeholder(tf.float32, shape = [BATCH_SIZE,NUM_CLASSES],name = 'labels')
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout_keep_prob')

    # Define multiple independent graphs
    l_logits = []
    for i in range(len(c.hidden_units)):
        with tf.variable_scope("RNN{0}".format(i)):
            # Create cell
            if len(c.hidden_units[i]) == 1:
                cell = create_cell(c, c.hidden_units[i][0], dropout_keep_prob)
            else:
                cell = tf.contrib.rnn.MultiRNNCell([create_cell(c,n,dropout_keep_prob) for n in c.hidden_units[i]])

            seq_len = get_seq_len(input_data) if c.dynamic_seq else None

            # Construct graph
            outputs, _ = tf.nn.dynamic_rnn(cell=cell, inputs=input_data, sequence_length=seq_len, dtype=c.dtype)

            # Extract last output
            last_output = outputs[:,-1,:]

            # Create extra feedforward layers if necessary
            for cu in c.extra_dense_units[i]:
                last_output = tf.layers.dense(last_output, cu,
                                              activation=c.extra_dense_activation,
                                              kernel_initializer=c.extra_dense_weights_initializer(),
                                              bias_initializer = c.extra_dense_bias_initializer())
                last_output = tf.contrib.layers.dropout(last_output, c.extra_dense_dropout_keep_prob)

            # Compute prediction linearly
            i_logits = tf.layers.dense(last_output, NUM_CLASSES)
        l_logits.append(i_logits)

    # Merge logits
    total = l_logits[0]
    for i in range(len(l_logits)-1):
        total = tf.add(total, l_logits[i+1])
    logits = tf.scalar_mul(tf.constant(1/len(l_logits), shape=(), dtype=c.dtype), total)

    # Eval
    correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'accuracy')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels), name = 'loss')
    optimizer = c.optimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss

if __name__ == "__main__":
    for a in define_graph():
        print(a)

    print(tf.train.AdamOptimizer())
