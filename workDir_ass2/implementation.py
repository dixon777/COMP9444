import tensorflow as tf
import sys, string, re
import numpy as np

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
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




l_configs = {
    "safe": {
        "learning_rate": 0.001
    }

}

class RNN(object):
    default_config = {
        "dtype": tf.float32,
        "learning_rate": 0.1,
        "l_hidden_units": np.array([[64,64],
                                    [64,64]]),
        "default_dropout_keep_prob": 0.75,
        "rnn_cell": tf.nn.rnn_cell.LSTMCell,
        "rnn_cell_intializer": tf.truncated_normal_initializer,
        "rnn_cell_activation": tf.nn.relu,
        "rnn_net": tf.nn.bidirectional_dynamic_rnn,
        "optimizer": tf.train.AdamOptimizer,
        "dynamic_seq": True
    }

    def __init__(self, input):
        for p in self.default_config:
            setattr(self, p, input.get(p,self.default_config[p]))

    @staticmethod
    def get_seq_len(m):
        a = tf.reduce_max(tf.abs(m), 2)
        one_or_zero = tf.sign(a)
        length = tf.cast(tf.reduce_sum(one_or_zero, 1), tf.int32)
        return length

    def create_cell(self, n, dropout_keep_prob):
        cell = self.rnn_cell(n, initializer=self.rnn_cell_intializer(),activation = self.rnn_cell_activation)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)



    def create_graph(self, inputs, labels, num_classes, dropout_keep_prob):
        seq_len = self.get_seq_len(inputs) if self.dynamic_seq else None
        if self.rnn_net == tf.contrib.rnn.stack_bidirectional_rnn:
            # self.l_hidden_units = [[fw_units_1, fw_units_2, ...], [bw_units_1, bw_units_2, ...]]
            l_hidden_units = l_hidden_units if len(np.shape(l_hidden_units)) > 2 else l_hidden_units[0, :]
            cell_fw = [self.create_cell(n, dropout_keep_prob) for n in self.l_hidden_units[0]]
            cell_bw = [self.create_cell(n, dropout_keep_prob) for n in self.l_hidden_units[1]]
            (value_fw,value_bw), _ = self.rnn_net(cell_fw=cell_fw, cell_bw=cell_bw, dtype=self.dtype, inputs=inputs, sequence_length=seq_len)
            outputs = tf.concat((value_fw, value_bw),2)

        elif self.rnn_net == tf.nn.bidirectional_dynamic_rnn:
            # self.l_hidden_units = [fw_units, bw_units]
            hidden = self.l_hidden_units
            while len(np.shape(hidden)) > 1:
                hidden = hidden[0]

            cell_fw = self.create_cell(hidden[0], dropout_keep_prob)
            cell_bw = self.create_cell(hidden[1], dropout_keep_prob)
            print("AAA")
            (value_fw,value_bw), _ = self.rnn_net(cell_fw=cell_fw, cell_bw=cell_bw, dtype=self.dtype, inputs=inputs, sequence_length=seq_len)
            print("AAA")
            outputs = tf.concat((value_fw, value_bw),2)
        elif self.rnn_net == tf.nn.dynamic_rnn: # Forward RNN
            # self.l_hidden_units = [units_1, units_2, ...]
            hidden = self.l_hidden_units
            while len(np.shape(hidden)) > 1:
                hidden = hidden[0]

            cell = create_cell(hidden, dropout_keep_prob) if len(hidden)==0 else tf.nn.rnn_cell.MultiRNNCell([self.create_cell(n, dropout_keep_prob)  for n in hidden])
            outputs, states = self.rnn_net(cell, dtype =self.dtype, inputs=inputs, sequence_length=seq_len)
        else:
            raise AttributeError("\"rnn_net\" is not either, tf.contrib.nn.stack_bidirectional_rnn, tf.nn.bidirectional_dynamic_rnn or tf.nn.dynamic_rnn")

        last_output = outputs[:,-1,:]
        logits = tf.layers.dense(last_output, num_classes)

        correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
        Accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'accuracy')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels), name = 'loss')
        optimizer = self.optimizer().minimize(loss)
        return optimizer, Accuracy, loss


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
    config = l_configs["safe"]

    # Define placeholders
    input_data = tf.placeholder(tf.float32,shape=[BATCH_SIZE,MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE],name='input_data')
    labels = tf.placeholder(tf.float32, shape = [BATCH_SIZE,NUM_CLASSES],name = 'labels')
    dropout_keep_prob = tf.placeholder_with_default(0.75, shape=(), name='dropout_keep_prob')

    # Obtain results
    rnn = RNN(config)
    optimizer, Accuracy, loss = rnn.create_graph(input_data, labels, NUM_CLASSES, dropout_keep_prob)
    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss

if __name__ == "__main__":
    a = define_graph()
    for i in a:
        print(type(i))
