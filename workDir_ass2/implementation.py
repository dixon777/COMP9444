import tensorflow as tf
import sys, string, re
BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

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

# Additional helper functions
# 08/09/2018 create
# 14/09/2018 separate into two func and make syntax more compact
def create_lstmCell(n, dropout_keep_prob):
    cell = tf.nn.rnn_cell.LSTMCell(n, initializer=tf.truncated_normal_initializer(),activation = tf.nn.relu)
    # if dropout_keep_prob.input>=1:
    # return cell
    # else:
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    # return cell if dropout_keep_prob>=1 else tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

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
    learning_rate = 0.1   # learning rate
    hidden_layer_units = [64,64]        # Define hidden layers
    lstmUnits = 64
    input_data = tf.placeholder(tf.float32,shape= [BATCH_SIZE,MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE],name='input_data')
    labels = tf.placeholder(tf.float32, shape = [BATCH_SIZE,2],name = 'labels')
    dropout_keep_prob = tf.placeholder_with_default(0.75, shape=(), name='dropout_keep_prob')


    lstm_fw_cell = tf.contrib.rnn.LSTMCell(lstmUnits,forget_bias= 1.0, initializer = tf.orthogonal_initializer())
    lstm_bw_cell = tf.contrib.rnn.LSTMCell(lstmUnits,forget_bias= 1.0, initializer = tf.orthogonal_initializer())
    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=dropout_keep_prob)
    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=dropout_keep_prob)
    #outputs,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell, x, dtype = tf.float32)
        
    (value_fw,value_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw =lstm_fw_cell, cell_bw =  lstm_bw_cell, dtype = tf.float32, inputs = input_data)
    # multi_cell = create_multiRNNcell(hidden_layer_units,dropout_keep_prob)
    # outputs, _ = tf.nn.dynamic_rnn(multi_cell, input_data,dtype=tf.float32)
    value = tf.concat((value_fw, value_bw),2)
    last_output = value[:,-1,:]

    prediction = tf.layers.dense(last_output,2)

    # see the accuracy
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    Accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name = 'accuracy')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels), name = 'loss')
    optimizer = tf.train.AdamOptimizer().minimize(loss)




    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
