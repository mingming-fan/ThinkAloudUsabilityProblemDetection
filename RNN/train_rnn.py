import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import time
from rnn import RNN
import data_helpers
import json as jsn
import pickle
import random
from tensorflow.contrib import learn
import nltk.corpus
import string
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC


# Parameters
# ==================================================
# Model Hyperparameters
tf.flags.DEFINE_string("cell_type", "gru", "Type of rnn cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (Default: 300)")
tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of character embedding (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

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

                    ## for IIR purpose, coder2 doesn't have sentiment value while coder1 does,
                    ## thus ban sentiment value in order to be consistent.
                    # sentiment[type].append(elem['sentiment_gt'])

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


def remove_stopword(data_set):
    print("removing stop words...")
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.extend(string.punctuation)
    stopwords.append('')
    for i in range(len(data_set)):
        sentence = data_set[i].split()
        remove = [word for word in sentence if word not in stopwords]
        s = ''
        for j in remove:
            s += j + ' '
        data_set[i] = s[:-1]
    return data_set


def cleanup(data, target, category):
    # delete category 'Explanation' for new since not informative.
    new = list(zip(data, target))
    new_mix = [x for x in new if x[1][categories.index(category)] != 1 and x[0] != '']
    data = [x[0] for x in new_mix]
    target = [x[1] for x in new_mix]

    return data, target


def compute_recall(prediction, target):
    if len(prediction) != len(target):
        print('compute recall error!')
        return 0

    cnt = 0
    total = 0
    for i in range(len(prediction)):
        # should be a problem
        if target[i][1] == 1:
            total += 1
            # predict as probelm
            if prediction[i] == 1:
                cnt += 1
    # no problem in ground truth targets
    if total == 0:
        return 0
    print("problem: {}, detect: {}".format(total, cnt))
    return cnt / total


def compute_accuracy_precision_recall(prediction, target, foldID, evalFile):
    if len(prediction) != len(target):
        print('compute recall error!')
        return 0

    accuracy = 0
    recall_cnt = 0
    recall_total = 0
    precision_cnt = 0
    precision_total = 0
    acc_cnt = 0

    for i in range(len(prediction)):
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
    return accuracy, precision, recall


def train(x_train, y_train, vocab_processor, x_dev, y_dev, foldID, evalFile):
    print("enter training")
    result = {'acc': [], 'precision': [], 'recall': [], 'prediction': []}
    with tf.device('/gpu:0'):
        x_train = x_train
        y_train = y_train
        x_dev = x_dev
        y_dev = y_dev

    with tf.Graph().as_default():
        print("start RNN init.")
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        print("finish RNN init.")
        with sess.as_default():
            rnn = RNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                cell_type=FLAGS.cell_type,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(rnn.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
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
            vocab_processor.save(os.path.join(out_dir, "text_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...

            # avoid using variable name that is the same as the function name. i.e. len
            i = 0
            for batch in batches:
                i += 1
                x_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.input_y: y_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, prediction, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, rnn.predictions, rnn.loss, rnn.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    prediction = prediction.tolist()
                    # recall = compute_recall(prediction, y_batch)
                    # print("{}: step {}, loss {:g}, acc {:g}, recall {:g}".format(time_str, step, loss, accuracy, recall))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    # only print the result of the evaluation when the model is trained to the last batch
                    print("\nEvaluation:")
                    feed_dict_dev = {
                        rnn.input_text: x_dev,
                        rnn.input_y: y_dev,
                        rnn.dropout_keep_prob: 1.0
                    }
                    summaries_dev, prediction, loss, accuracy = sess.run(
                        [dev_summary_op, rnn.predictions, rnn.loss, rnn.accuracy], feed_dict_dev)
                    dev_summary_writer.add_summary(summaries_dev, step)

                    time_str = datetime.datetime.now().isoformat()
                    prediction = prediction.tolist()

                    # print("{}: step {}, loss {:g}, acc {:g} recall {:g}\n".format(time_str, step, loss, accuracy, recall))

                    accuracy, precision, recall = compute_accuracy_precision_recall(prediction, y_dev, foldID, evalFile)

                    result['acc'].append(accuracy)
                    result['precision'].append(precision)
                    result['recall'].append(recall)
                    result['prediction'].append(prediction)

                # detect if early stop is needed
                if len(result['recall']) >= 3 and (
                                    result['recall'][-1] <= result['recall'][-2] <= result['recall'][-3] or
                                    result['precision'][-1] <= result['precision'][-2] <= result['precision'][-3]):
                    print("start overfiting, early stop occurs")
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))
                    break
                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

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


if __name__ == "__main__":
    dir = './user-data'
    data, target, category, sentiment, pitch, speechrate, \
    device_data, device_target, website_data, website_target = reload_data()

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

    # vectorize sentences
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
    data = np.append(data, speechrate_full, axis=1)

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
    rnnEvaluationResults = "rnn_{:d}_fold_cv_evalution.csv".format(x_fold)
    evaluationFile = open(dir + '/' + rnnEvaluationResults, 'w')
    kf = KFold(n_splits=x_fold)
    foldID = 1
    overall_acc = 0
    overall_recall = 0
    overall_precision = 0
    print('10 fold cross validation:\n')
    for train_index, test_index in kf.split(data):
        print('fold ID: {:d}\n'.format(foldID))
        train_data, test_data = data_pure[train_index], data_pure[test_index]
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

        acc, recall, precision, test_predict = train(tf_idf_train, train_target, vocab_processor, tf_idf_test_autoCategory, test_target, foldID,
                                       evaluationFile)
        acc, precision, recall = compute_accuracy_precision_recall(test_predict, test_target, foldID, evaluationFile)

        foldID += 1
        overall_acc += acc
        overall_precision += precision
        overall_recall += recall

    evaluationFile.close()

    acc = overall_acc / x_fold
    recall = overall_recall / x_fold
    precision = overall_precision / x_fold
    F1Score = 2 * recall * precision / (recall + precision)

    print("\nRNN: accuracy: {}, precision: {}, recall: {}, F1Score: {}\n".format(
            acc, precision, recall, F1Score))