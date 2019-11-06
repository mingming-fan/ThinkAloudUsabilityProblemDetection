import os
import time
import datetime as dt
import data_helpers
import json as jsn
import pickle
import random
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.contrib import learn
from textCNN import TextCNN
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
from pandas import *
from sklearn.model_selection import KFold
import copy
import sklearn.metrics as sklmetric
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# Parameters
# ==================================================
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

categories = ['Observation', 'Procedure', 'Explanation', 'Reading']
category_map = {'Observation': [0, 0], 'Procedure': [0, 1], 'Explanation': [1, 0], 'Reading': [1, 1]}


# load raw data from the current directory
def load_data(dir, categories):
    cnt = 0
    data = {'digital-website': [], 'physical-device': []}
    target = {'digital-website': [], 'physical-device': []}
    category = {'digital-website': [], 'physical-device': []}
    sentiment = {'digital-website': [], 'physical-device': []}

    # load pitch data for each sentence for each file
    pitch = {'digital-website': [], 'physical-device': []}

    # load speech rate data for each sentence for each file
    speechrate = {'digital-website': [], 'physical-device': []}

    for type in ['digital-website', 'physical-device']:
        file_lists = os.listdir(dir + '/' + type + '/')
        for file in file_lists:
            if file.split('.')[-1] == 'json':
                print(file)
                f = open(dir + '/' + type + '/' + file, 'r')
                collection = jsn.load(f)
                cnt += len(collection)

                pitch_collection_perFile = []
                speechrate_collection_perFile = []

                # in terms of problem, convert label to one-hot vector
                for elem in collection:
                    data[type].append(elem['transcription'])
                    if elem['problem'] == 1:
                        label = [0, 1]
                    else:
                        label = [1, 0]
                    target[type].append(label)

                    category_code = category_map[elem['category']]
                    category[type].append(category_code)

                    sentiment[type].append(elem['sentiment_gt'])

                    # add the list of pitches of current sentence to pitch collection
                    pitch_collection_perFile.extend(elem['pitch'])
                    speechrate_collection_perFile.append(elem['speechrate'])

                # compute the avg and std of the pitch within a file scope
                avg_pitch = np.mean(pitch_collection_perFile)
                std_pitch = np.std(pitch_collection_perFile)

                # compute the avg and std of the speech rate within a file scope
                avg_speechrate = np.mean(speechrate_collection_perFile)
                std_speechrate = np.std(speechrate_collection_perFile)

                # computation for each sentence in current file
                # to determine if should have an abnormal pitch and abnormal speech rate.
                for i in range(len(collection)):
                    # abnormal pitch
                    seg_pitch = collection[i]['pitch']
                    abnormal_cnt = []
                    for p in seg_pitch:
                        if (p - avg_pitch) >= 2 * std_pitch:
                            abnormal_cnt.append(1)
                        if (p - avg_pitch) <= -2 * std_pitch:
                            abnormal_cnt.append(-1)
                        else:
                            abnormal_cnt.append(0)
                    # check if there're K consecutive abnormal pitch in one segment
                    pitch_high_threshold = 3
                    pitch_low_threshold = 3
                    pitch_code = [0, 0]  # [if_high_pitch, if_low_pitch]
                    if abnormal_cnt.count(1) >= pitch_high_threshold:
                        pitch_code[0] = 1
                    if abnormal_cnt.count(-1) >= pitch_low_threshold:
                        pitch_code[1] = 1
                    pitch[type].append(pitch_code)
                    collection[i]['abnormal_pitch'] = pitch[type][-1]

                    # abnormal speech rate
                    seg_speechrate = collection[i]['speechrate']
                    speechrate_code = [0, 0]  # [if high_speechrate, if_low_speechrate]
                    if (seg_speechrate - avg_speechrate) >= 2 * std_speechrate:
                        speechrate_code[0] = 1
                    if (seg_speechrate - avg_speechrate) <= -2 * std_speechrate:
                        speechrate_code[1] = 1
                    speechrate[type].append(speechrate_code)
                    collection[i]['abnormal_speechrate'] = speechrate[type][-1]

                f.close()
    print("{0} sentences in total".format(cnt))
    return data, target, category, sentiment, pitch, speechrate


# extract features and assign weights
def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data)  # bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names()  # converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data)
    return tf_idf_train, tf_idf_test, feature_names


def reload_data():
    print("enter reload...\n")
    # data and target initialization
    dir = './user-data'
    categories = ['Observation', 'Procedure', 'Explanation', 'Reading']
    # no previous loaded data, load and save them
    if not (os.path.isfile('./save-data/digital-website/data.txt')
            and os.path.isfile('./save-data/digital-website/target.txt')
            and os.path.isfile('./save-data/digital-website/category.txt')
            and os.path.isfile('./save-data/physical-device/data.txt')
            and os.path.isfile('./save-data/physical-device/target.txt')
            and os.path.isfile('./save-data/physical-device/category.txt')
            and os.path.isfile('./save-data/physical-device/sentiment.txt')
            and os.path.isfile('./save-data/digital-website/sentiment.txt')
            and os.path.isfile('./save-data/physical-device/pitch.txt')
            and os.path.isfile('./save-data/digital-website/pitch.txt')
            and os.path.isfile('./save-data/physical-device/speechrate.txt')
            and os.path.isfile('./save-data/digital-website/speechrate.txt')
            ):
        print("load data")
        data, target, category, sentiment, pitch, speechrate = load_data(dir, categories)

        website_speechrate  = speechrate ['digital-website']
        website_pitch = pitch['digital-website']
        website_sentiment = sentiment['digital-website']
        website_category = category['digital-website']
        website_data = data['digital-website']
        website_target = target['digital-website']

        device_speechrate = speechrate['physical-device']
        device_pitch = pitch['physical-device']
        device_sentiment = sentiment['physical-device']
        device_category = category['physical-device']
        device_data = data['physical-device']
        device_target = target['physical-device']

        with open("./save-data/digital-website/data.txt", "wb") as ddf:  # Pickling
            pickle.dump(data['digital-website'], ddf)
            ddf.close()
        with open("./save-data/digital-website/target.txt", "wb") as dtf:  # Pickling
            pickle.dump(target['digital-website'], dtf)
            dtf.close()
        with open("./save-data/digital-website/category.txt", "wb") as dcf:  # Pickling
            pickle.dump(category['digital-website'], dcf)
            dtf.close()
        with open("./save-data/digital-website/sentiment.txt", "wb") as dsf:  # Pickling
            pickle.dump(sentiment['digital-website'], dsf)
            dsf.close()
        with open("./save-data/digital-website/pitch.txt", "wb") as dpf:  # Pickling
            pickle.dump(pitch['digital-website'], dpf)
            dpf.close()
        with open("./save-data/digital-website/speechrate.txt", "wb") as dsrf:  # Pickling
            pickle.dump(speechrate['digital-website'], dsrf)
            dsrf.close()

        with open("./save-data/physical-device/data.txt", "wb") as pdf:  # Pickling
            pickle.dump(data['physical-device'], pdf)
            pdf.close()
        with open("./save-data/physical-device/target.txt", "wb") as ptf:  # Pickling
            pickle.dump(target['physical-device'], ptf)
            ptf.close()
        with open("./save-data/physical-device/category.txt", "wb") as pcf:  # Pickling
            pickle.dump(category['physical-device'], pcf)
            ptf.close()
        with open("./save-data/physical-device/sentiment.txt", "wb") as psf:  # Pickling
            pickle.dump(sentiment['physical-device'], psf)
            psf.close()
        with open("./save-data/physical-device/pitch.txt", "wb") as ppf:  # Pickling
            pickle.dump(pitch['physical-device'], ppf)
            ppf.close()
        with open("./save-data/physical-device/speechrate.txt", "wb") as psrf:  # Pickling
            pickle.dump(speechrate['physical-device'], psrf)
            psrf.close()

    # data already loaded, retrieve them from ./save-data
    else:
        print('retrieve data')

        with open("./save-data/digital-website/data.txt", "rb") as ddf:  # Pickling
            website_data = pickle.load(ddf)
            ddf.close()
        with open("./save-data/digital-website/target.txt", "rb") as dtf:  # Pickling
            website_target = pickle.load(dtf)
            dtf.close()
        with open("./save-data/digital-website/category.txt", "rb") as dcf:  # Pickling
            website_category = pickle.load(dcf)
            dtf.close()
        with open("./save-data/digital-website/sentiment.txt", "rb") as dsf:  # Pickling
            website_sentiment = pickle.load(dsf)
            dsf.close()
        with open("./save-data/digital-website/pitch.txt", "rb") as dpf:  # Pickling
            website_pitch = pickle.load(dpf)
            dpf.close()
        with open("./save-data/digital-website/speechrate.txt", "rb") as dsrf:  # Pickling
            website_speechrate = pickle.load(dsrf)
            dsrf.close()

        with open("./save-data/physical-device/data.txt", "rb") as pdf:  # Pickling
            device_data = pickle.load(pdf)
            pdf.close()
        with open("./save-data/physical-device/target.txt", "rb") as ptf:  # Pickling
            device_target = pickle.load(ptf)
            ptf.close()
        with open("./save-data/physical-device/category.txt", "rb") as pcf:  # Pickling
            device_category = pickle.load(pcf)
            ptf.close()
        with open("./save-data/physical-device/sentiment.txt", "rb") as psf:  # Pickling
            device_sentiment = pickle.load(psf)
            psf.close()
        with open("./save-data/physical-device/pitch.txt", "rb") as ppf:  # Pickling
            device_pitch = pickle.load(ppf)
            ppf.close()
        with open("./save-data/physical-device/speechrate.txt", "rb") as psrf:  # Pickling
            device_speechrate = pickle.load(psrf)
            psrf.close()

    # a union data set for mixture training and test
    data = device_data + website_data
    target = device_target + website_target
    category = device_category + website_category
    sentiment = device_sentiment + website_sentiment
    pitch = device_pitch + website_pitch
    speechrate = device_speechrate + website_speechrate

    for i in range(len(data)):
        # otherwise "coffee" and "coffee," would be treated as different tokens
        data[i] = data[i].replace(",", "")

        # split training and test data
    random.seed(3)
    random.shuffle(data)
    random.seed(3)
    random.shuffle(target)
    random.seed(3)
    random.shuffle(category)
    random.seed(3)
    random.shuffle(sentiment)
    random.seed(3)
    random.shuffle(pitch)
    random.seed(3)
    random.shuffle(speechrate)

    return data, target, category, sentiment, pitch, speechrate, device_data, device_target, website_data, website_target


def compute_accuracy_precision_recall(prediction, target, foldID, evalFile):
    if len(prediction) != len(target):
        print('compute recall error!')
        return 0
    precision = 0
    recall = 0
    accuracy = 0
    recall_cnt = 0
    recall_total = 0
    precision_cnt = 0
    precision_total = 0
    acc_cnt = 0

    # convert one hot target to 1 bit target
    one_bit_target = np.zeros(len(prediction))
    for i in range(len(prediction)):
        if target[i][1] == 1:
            one_bit_target[i] = 1
        if prediction[i] == target[i][1]:
            acc_cnt += 1
        # for precision
        if prediction[i] == 1:
            precision_total += 1
            if target[i][1] == 1:
                precision_cnt += 1
        # should be a problem
        if target[i][1] == 1:
            recall_total += 1
            # predict as probelm
            if prediction[i] == 1:
                recall_cnt += 1

    if len(prediction) > 0:
        accuracy = acc_cnt / len(prediction)

    # no problem in ground truth targets
    if recall_total == 0:
        recall = 0
    else:
        recall = recall_cnt / recall_total

    if precision_total == 0:
        precision = 0
    else:
        precision = precision_cnt / precision_total
    evalFile.write("{:g},{:g},{:g},{:g}\n".format(foldID, accuracy, precision, recall))
    print("\nfoldID: {}, accuracy: {}, precision: {}, recall: {}\n".format(foldID, accuracy, precision, recall))
    return accuracy, precision, recall, one_bit_target


def get_false_pos_neg(prediction, target):
    """ return the index of false negative and false positive sentences. """

    false_positive = []
    false_negative = []
    for i in range(len(prediction)):
        # wrong prediction
        if prediction[i] != target[i][1]:
            # predict it as a non-problem, false negative
            if prediction[i] == 0:
                false_negative.append(i)

            # predict it as a problem, false positive
            if prediction[i] == 1:
                false_positive.append(i)

    return false_positive, false_negative


def train(x_train, y_train, vocab_processor, x_dev, y_dev, foldID, evaluationFile):
    result = {'acc': [], 'precision': [], 'recall': [], 'prediction': []}
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                _, step, summaries, prediction, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.predictions, cnn.loss, cnn.accuracy],
                    feed_dict)

                time_str = dt.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, prediction, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.predictions, cnn.loss, cnn.accuracy],
                    feed_dict)

                # generate confusion matrix
                prediction = prediction.tolist()
                target = []
                for instance in y_batch:
                    target.append(instance.tolist().index(1))
                accuracy, precision, recall, one_bit_target = compute_accuracy_precision_recall(prediction, y_batch, foldID,
                                                                                evaluationFile)

                result['acc'].append(accuracy)
                result['precision'].append(precision)
                result['recall'].append(recall)
                result['prediction'].append(prediction)

                time_str = dt.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")

                # detect if early stop is needed
                if len(result['recall']) >= 3 and (
                                    result['recall'][-1] <= result['recall'][-2] <= result['recall'][-3] or
                                    result['precision'][-1] <= result['precision'][-2] <= result['precision'][-3]):
                    print("start overfiting, early stop occurs")
                    break

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            # when base line result needs to be returned(i.e. for voting)
            return result['acc'][-3], result['recall'][-3], result['precision'][-3], result['prediction'][-3]


def negation_idx(data, negative_words):
    negation_idx = np.array([[0] for i in range(len(data))])
    for i in range(len(data)):
        sentence = data[i].split()
        for word in sentence:
            if word in negative_words:
                negation_idx[i] = 1
                break
    return negation_idx


def category_idx(data, category):
    if len(data) != len(category):
        print("error in category index!")
    category_idx = np.array([[0, 0] for i in range(len(data))])
    for i in range(len(data)):
        category_idx[i] = category[i]
    return category_idx


def sentiment_idx(data, sentiment):
    if len(data) != len(sentiment):
        print("error in category index!")
    sentiment_idx = np.array([[0] for i in range(len(data))])
    for i in range(len(data)):
        # add 1 since NN model won't accept negative value
        sentiment_value = sentiment[i] + 1
        sentiment_idx[i] = sentiment_value
    return sentiment_idx


def interrogative_idx(data, interrogative_words):
    interrogative_idx = np.array([[0] for i in range(len(data))])
    for i in range(len(data)):
        sentence = data[i].split()
        for word in sentence:
            if word in interrogative_words:
                interrogative_idx[i] = 1
                break
    return interrogative_idx


def pitch_idx(data, pitch):
    if len(data) != len(pitch):
        print("error in pitch index!")
    pitch_idx = np.array([pitch[i] for i in range(len(data))])

    return pitch_idx


def speechrate_idx(data, speechrate):
    if len(data) != len(speechrate):
        print("error in speechrate index!")
    speechrate_idx = np.array([speechrate[i] for i in range(len(data))])

    return speechrate_idx


if __name__ == '__main__':
    print("enter main...")
    dir = './user-data'

    data, target, category, sentiment, pitch, speechrate, \
    device_data, device_target, website_data, website_target = reload_data()

    # deep copy sentences for false analysis
    data_sentences = copy.deepcopy(data)
    data_sentences = np.array(data_sentences)

    # ======== data preprocessing ======
    ## Build vocabulary
    # add negation information
    negative_words = ['no', 'not', 'don\'t', 'doesn\'t', 'didn\'t', 'never',
                      'but', 'nope', 'wasn\'t', 'won\'t', 'aren\'t', 'weren\'t',
                      'none', 'haven\'t', 'ain\'t', 'wouldn\'t']
    interrogative_words = ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
                           'What', 'When', 'Where', 'Which', 'Who', 'Whom', 'Whose', 'Why', 'How']

    negation_full = negation_idx(data, negative_words)
    category_full = category_idx(data, category)
    sentiment_full = sentiment_idx(data, sentiment)
    interrogative_full = interrogative_idx(data, interrogative_words)
    pitch_full = pitch_idx(data, pitch)
    speechrate_full = speechrate_idx(data, speechrate)

    # data augmentation: sentences + additional data
    # sentence vectorization
    max_document_length = max([len(x.split(" ")) for x in data])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    data = np.array(list(vocab_processor.fit_transform(data)))
    target = np.array(target)
    data_pure = np.copy(data)

    # data augmentation
    data = np.append(data, negation_full, axis=1)
    data = np.append(data, category_full, axis=1)
    data = np.append(data, sentiment_full, axis=1)
    data = np.append(data, interrogative_full, axis=1)
    data = np.append(data, pitch_full, axis=1)
    # data = np.append(data, speechrate_full, axis=1)


    # pure additional data augmentation
    # combine category and negation features
    features_only_input = np.append(category_full, negation_full, axis=1)

    # add interrogation to the end of matrix
    features_only_input = np.append(features_only_input, interrogative_full, axis=1)

    # add sentiment to the end of matrix
    features_only_input = np.append(features_only_input, sentiment_full, axis=1)

    # add pitch to the end of matrix
    features_only_input = np.append(features_only_input, pitch_full, axis=1)

    # add speechrate to the end of matrix
    features_only_input = np.append(features_only_input, speechrate_full, axis=1)

    # 10 fold cross validation
    x_fold = 10
    cnnEvaluationResults = "cnn_{:d}_fold_cv_evalution.csv".format(x_fold)
    evaluationFile = open(dir + '/' + cnnEvaluationResults, 'w')
    kf = KFold(n_splits=x_fold)
    foldID = 1
    overall_acc = 0
    overall_recall = 0
    overall_precision = 0

    print('10 fold cross validation:\n')
    for train_index, test_index in kf.split(data):

        print('fold ID: {:d}\n'.format(foldID))
        train_data, test_data = data[train_index], data[test_index]
        train_target, test_target = target[train_index], target[test_index]

        test_neagtion_info = negation_full[test_index]
        test_category_info = category_full[test_index]
        test_sentiment_info = sentiment_full[test_index]
        test_interrogative_info = interrogative_full[test_index]
        test_pitch_info = pitch_full[test_index]
        test_speechrate_info = speechrate_full[test_index]

        # ========================== STRAT =========================
        # ======= automatically generate test category info ========
        category_train_y = []
        for i in range(len(category_full[train_index])):
            if np.array_equal(category_full[train_index][i], [0, 0]):
                category_train_y.append(0)
            elif np.array_equal(category_full[train_index][i], [0, 1]):
                category_train_y.append(1)
            elif np.array_equal(category_full[train_index][i], [1, 0]):
                category_train_y.append(2)
            elif np.array_equal(category_full[train_index][i], [1, 1]):
                category_train_y.append(3)
            else:
                print("error, 5th category occurs")
        category_train_y = np.array(category_train_y)
        svm_category = LinearSVC(C=0.5)
        svm_category.fit(data_pure[train_index], category_train_y)
        auto_category = svm_category.predict(data_pure[test_index])

        auto_category_info = []
        for i in range(len(auto_category)):
            if auto_category[i] == 0:
                auto_category_info.append([0, 0])
            elif auto_category[i] == 1:
                auto_category_info.append([0, 1])
            elif auto_category[i] == 2:
                auto_category_info.append([1, 0])
            elif auto_category[i] == 3:
                auto_category_info.append([1, 1])
            else:
                print(auto_category[i])
                print("error, 5th category occurs")
        auto_category_info = np.array(auto_category_info)
        train_category_info = category_full[train_index]

        # generate training and test data(auto category)
        tf_idf_test_autoCategory = np.append(data_pure[test_index], auto_category_info, axis=1)
        tf_idf_train = np.append(data_pure[train_index], train_category_info, axis=1)

        # ======= automatically generate test category info ========
        # ======================== END =============================

        # augmentation prediction
        acc, recall, precision, test_predict = train(tf_idf_train, train_target, vocab_processor,
                                                     tf_idf_test_autoCategory, test_target, foldID, evaluationFile)

        acc, precision, recall, one_bit_target = compute_accuracy_precision_recall(test_predict, test_target, foldID, evaluationFile)

        foldID += 1
        overall_acc += acc
        overall_precision += precision
        overall_recall += recall

    evaluationFile.close()

    acc = overall_acc / x_fold
    recall = overall_recall / x_fold
    precision = overall_precision / x_fold
    F1Score = 2 * recall * precision / (recall + precision)

    print("\nCNN: accuracy: {}, precision: {}, recall: {}, F1Score: {}\n".format(
        acc, precision, recall, F1Score))
