import tensorflow as tf
import sys, string, re

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
# num_classes = 2

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

def get_accuracy_definition(preds_op, labels):
    correct_preds_op = tf.equal(tf.argmax(preds_op, 1), tf.argmax(labels, 1))
    # the tf.cast sets True to 1.0, and False to 0.0. With N predictions, of
    # which M are correct, the mean will be M/N, i.e. the accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32), name="accuracy")
    return accuracy

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

# Additional helper functions
# 08/09/2018 create
# 14/09/2018 separate into two func and make syntax more compact
def create_lstmCell(n, dropout_keep_prob):
    cell = tf.nn.rnn_cell.LSTMCell(n, initializer=tf.truncated_normal_initializer(),activation = tf.nn.relu)
    return cell if dropout_keep_prob>=1 else tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

def create_multiRNNcell(hidden,dropout_keep_prob):
    cells = [create_lstmCell(n, dropout_keep_prob) for n in hidden]
    multiCell = tf.nn.rnn_cell.MultiRNNCell(cells)
    return multiCell




def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    # Hyperparameters definition
    learning_rate = 0.001   # learning rate
    hidden_layer_units = [128,128]        # Define hidden layers

    # Input placeholder
    input_data = tf.placeholder(tf.float32,shape= [BATCH_SIZE,MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE],name='input_data')
    labels = tf.placeholder(tf.float32, shape = [BATCH_SIZE,2],name = 'labels')
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name='dropout_keep_prob')


    multi_cell = define_multi_cell(hidden_layer_units,dropout_keep_prob)
    outputs, _ = tf.nn.dynamic_rnn(multi_cell, input_data,dtype=tf.float32)
    # Note: size(outputs) = [BATCH_SIZE, MAX_WORDS_IN_REVIEW, hidden_layer_units[-1]]

    # Extract last outputs
    # Note: size(last_output) = [BATCH_SIZE, hidden_layer_units[-1]]
    # Option 1:
    # transposed_outputs = tf.transpose(outputs,[1,0,2])
    # last_output = tf.gather(outputs, int(tf.shape(transposed_outputs)[0]) - 1)
    # Option 2:
    last_output = tf.slice(outputs, [0, int(tf.shape(outputs)[1]) - 1, 0], [BATCH_SIZE, 1, EMBEDDING_SIZE])


    # Do last computation for last transpose
    weights = tf.Variable(tf.truncated_normal([hidden_layer_units[-1], 2]))
    biases = tf.Variable(tf.random_normal([2]))
    logits = tf.matmul(last_output, weights) + biases

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits), name="loss")

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

    # Evaluate model
    pred_op = tf.nn.softmax(logits)
    Accuracy = get_accuracy_definition(pred_op, labels, name="accuracy")









    # global_step = tf.Variable(0)
    # #learning_rate = tf.train.exponential_decay(0.2, global_step, 5000, 0.9, staircase=True)
    # optimizer = tf.train.GradientDescentOptimizer(0.1)
    # last_elem = trans_rnn[-1]
    # logits = (tf.matmul(last_elem, W) + b)
    # # make prediction and calculate the loss
    # prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    # accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name="accuracy")
    # losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    # loss = tf.reduce_mean(losses, name="loss")


    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
