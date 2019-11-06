from __future__ import division
import os
import json as JSON
from collections import OrderedDict
import pickle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
from pandas import *
from sklearn.model_selection import KFold
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
import argparse

warnings.filterwarnings("ignore")

# the total number of users
users = 8


def load_label(dir):
    print("load label: {}".format(dir))
    users = 8

    label = {}
    # label = {"machine": {"user1": [0,1,0,1,1,...,0], "user2": [0,1,0,1,1,...,0]}, ...},
    #          "history": {"user1": [0,1,0,1,1,...,0], "user2": [0,1,0,1,1,...,0]}, ...},
    #          "remote": {"user1": [0,1,0,1,1,...,0], "user2": [0,1,0,1,1,...,0]}, ...},
    #          "science": {"user1": [0,1,0,1,1,...,0], "user2": [0,1,0,1,1,...,0]}
    #          }
    for product in products:
        label[product] = {}
        for user_num in range(1, users+1):
            user = "user" + str(user_num)
            label[product][user] = []

    for product in products:
        for user_num in range(1, users+1):
            user = "user"+str(user_num)
            file_lists = os.listdir(dir+'/'+product+'/'+user+'/')
            for file in file_lists:
                if file.split('.')[-1] == 'json':
                    print(dir+'/'+product+'/'+user+'/'+file)

                    with open(dir+'/'+product+'/'+user+'/'+file, 'r') as f:
                        collection = JSON.load(f)
                        for elem in collection:
                            label[product][user].append(elem['problem'])
                        f.close()

    return label


# load raw data from the current directory
def load_data(dir):
    cnt = 0
    features = ["transcript", "category", "sentiment", "pitch", "speechrate", "index"]
    data = {}
    target = {}
    users = 8

    for product in products:
        target[product] = {}
        data[product] = {}
        # data = {"machine":
        #           {"user1": {"transcript": [], "sentiment": [], "pitch": [], "speechrate": [], "category: []"},
        #            "user2": {"transcript": [], "sentiment": [], "pitch": [], "speechrate": [], "category: []"},
        #             ...
        #            "user8": {"transcript": [], "sentiment": [], "pitch": [], "speechrate": [], "category: []"}
        #           },
        #         ...
        #         "science":
        #           {"user1": {"transcript": [], "sentiment": [], "pitch": [], "speechrate": [], "category: []"},
        #            "user2": {"transcript": [], "sentiment": [], "pitch": [], "speechrate": [], "category: []"},
        #             ...
        #            "user8": {"transcript": [], "sentiment": [], "pitch": [], "speechrate": [], "category: []"}
        #           }
        #        }
        for user_num in range(1, users+1):
            user = "user" + str(user_num)
            target[product][user] = []
            data[product][user] = {}
            for feature in features:
                data[product][user][feature] = []

    for product in products:
        for user_num in range(1, users+1):
            user = "user" + str(user_num)
            file_lists = os.listdir(dir+'/'+product+'/'+user+'/')
            for file in file_lists:
                if file.split('.')[-1] == 'json':
                    print(dir+'/'+product+'/'+user+'/'+file)

                    with open(dir+'/'+product+'/'+user+'/'+file, 'r') as f:
                        collection = JSON.load(f, object_pairs_hook=OrderedDict)
                        cnt += len(collection)
                        pitch_collection_perFile = []
                        speechrate_collection_perFile = []
                        loudness_collection_perFile = []
                        for elem in collection:
                            data[product][user]["transcript"].append(elem['transcription'])

                            idx = elem['problem']
                            target[product][user].append(idx)

                            category_code = category_map[elem['category']]
                            data[product][user]["category"].append(category_code)

                            data[product][user]["sentiment"].append(elem['sentiment_gt'])

                            # add the "list" of pitches of current sentence to pitch collection
                            pitch_collection_perFile.extend(elem['pitch'])
                            # append the speech rate of current sentence to speech rate collection
                            speechrate_collection_perFile.append(elem['speechrate'])
                            # add the "list" of loudness of current sentence to loudness collection
                            seg_loudness = [float(num) for num in elem['loudness']]
                            loudness_collection_perFile.extend(seg_loudness)

                        # compute the avg and std of the pitch within a file scope
                        avg_pitch = np.mean(pitch_collection_perFile)
                        std_pitch = np.std(pitch_collection_perFile)
                        min_pitch = np.min(pitch_collection_perFile)
                        max_pitch = np.max(pitch_collection_perFile)

                        # compute the avg and std of the speech rate within a file scope
                        avg_speechrate = np.mean(speechrate_collection_perFile)
                        std_speechrate = np.std(speechrate_collection_perFile)
                        min_speechrate = np.min(speechrate_collection_perFile)
                        max_speechrate = np.max(speechrate_collection_perFile)

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
                            data[product][user]["pitch"].append(pitch_code)


                            # abnormal speech rate
                            seg_speechrate = collection[i]['speechrate']
                            speechrate_code = [0, 0]  # [if high_speechrate, if_low_speechrate]
                            if (seg_speechrate - avg_speechrate) >= 2 * std_speechrate:
                                speechrate_code[0] = 1
                            if (seg_speechrate - avg_speechrate) <= -2 * std_speechrate:
                                speechrate_code[1] = 1
                            data[product][user]["speechrate"].append(speechrate_code)

                        f.close()

    print("{0} sentences in total".format(cnt))
    return data, target


tokenize = lambda doc: doc.lower().split(" ")


def compute_recall(prediction, target):
    if len(prediction) != len(target):
        print('compute recall error!')
        return 0
    cnt = 0
    total = 0
    for i in range(len(prediction)):
        # should be a problem
        if target[i] == 1:
            total += 1
            # predict as problem
            if prediction[i] == 1:
                cnt += 1
    # no problem in ground truth targets
    if total == 0:
        return 0
    print("truth positive: {}, detect: {} among the ground truths".format(total, cnt))
    return cnt/total


def compute_precision(prediction, target):
    if len(prediction) != len(target):
        print('compute recall error!')
        return 0
    cnt = 0
    total = 0
    for i in range(len(prediction)):
        # predict as a problem
        if prediction[i] == 1:
            total += 1
            # indeed a probelm
            if target[i] == 1:
                cnt += 1
    # no problem in ground truth targets
    if total == 0:
        return 0
    print("prediction: {}, {} of which are truth positive's ".format(total, cnt))
    return cnt/total


def compute_accuracy(prediction, target):
    cnt = 0
    for i in range(target.shape[0]):
        if prediction[i] == target[i]:
            cnt += 1
    return cnt / target.shape[0]


def compute_f1score(prediction, target):
    precision = compute_precision(prediction, target)
    recall = compute_recall(prediction, target)

    print("pre={} recall={}".format(precision, recall))
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def negation_idx(data, negative_words):
    negation_idx = np.array([[0] for i in range(len(data))])
    for i in range(len(data)):
        sentence = data[i].split()
        for word in sentence:
            if word in negative_words:
                negation_idx[i] = 1
                break
    return negation_idx


def interrogative_idx(data, interrogative_words):
    interrogative_idx = np.array([[0] for i in range(len(data))])
    for i in range(len(data)):
        sentence = data[i].split()
        for word in sentence:
            if word in interrogative_words:
                interrogative_idx[i] = 1
                break
    return interrogative_idx


def category_idx(data, category):
    if len(data) != len(category):
        print("error in category index!")
    category_idx = np.array([category[i] for i in range(len(data))])

    return category_idx


def sentiment_idx(data, sentiment):
    if len(data) != len(sentiment):
        print("error in sentiment index!")
    sentiment_idx = np.array([[sentiment[i]] for i in range(len(data))])

    return sentiment_idx


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


category_map = {'Observation': [0, 0], 'Procedure': [0, 1], 'Explanation': [1, 0], 'Reading': [1, 1]}
negative_words = ['no', 'not', 'don\'t', 'doesn\'t', 'didn\'t', 'never',
                      'but', 'nope', 'wasn\'t', 'won\'t', 'aren\'t', 'weren\'t',
                      'none', 'haven\'t', 'ain\'t', 'wouldn\'t']

interrogative_words = ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
                        'What', 'When', 'Where', 'Which', 'Who', 'Whom', 'Whose', 'Why', 'How']

products = ["machine", "history", "remote", "science"]

# extract features and assign weights
def tf_idf_features(train_data, test_data, train_category, test_category,
                    train_sentiment, test_sentiment,
                    train_pitch, test_pitch,
                    train_speechrate, test_speechrate
                    ):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize, stop_words=['a','an','the','I']) #stop_words=None #stop_words=text.ENGLISH_STOP_WORDS
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data)

    negative_words = ['no', 'not', 'don\'t', 'doesn\'t', 'didn\'t', 'never',
                      'but', 'nope', 'wasn\'t', 'won\'t', 'aren\'t', 'weren\'t',
                      'none', 'haven\'t', 'ain\'t', 'wouldn\'t']

    interrogative_words = ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
                           'What', 'When', 'Where', 'Which', 'Who', 'Whom', 'Whose', 'Why', 'How']

    # training data： negation augmentation
    train_negation_index = negation_idx(train_data, negative_words)

    #  training data：category augmentation
    train_category_index = category_idx(train_data, train_category)

    # training data: interrogation augmentation
    train_interrogation_index = interrogative_idx(train_data, interrogative_words)

    # training data: sentiment augmentation
    train_sentiment_index = sentiment_idx(train_data, train_sentiment)

    # training data: pitch augmentation
    train_pitch_index = pitch_idx(train_data, train_pitch)

    # training data: speechrate augmentation
    train_speechrate_index = speechrate_idx(train_data, train_speechrate)

    # convert sparse to array in order to do negation appending
    tf_idf_train = tf_idf_train.toarray()
    tf_idf_train = np.append(tf_idf_train, train_negation_index, axis=1)
    tf_idf_train = np.append(tf_idf_train, train_category_index, axis=1)
    tf_idf_train = np.append(tf_idf_train, train_interrogation_index, axis=1)
    tf_idf_train = np.append(tf_idf_train, train_sentiment_index, axis=1)
    tf_idf_train = np.append(tf_idf_train, train_pitch_index, axis=1)
    tf_idf_train = np.append(tf_idf_train, train_speechrate_index, axis=1)

    feature_names = tf_idf_vectorize.get_feature_names()  # converts feature index to the word it represents.

    tf_idf_test = tf_idf_vectorize.transform(test_data)

    # test data: negation augmentation
    test_negation_index = negation_idx(test_data, negative_words)

    # test data: category augmentation
    test_category_index = category_idx(test_data, test_category)

    # test data: interrogation augmentation
    test_interrogation_index = interrogative_idx(test_data, interrogative_words)

    # test data: sentiment augmentation
    test_sentiment_index = sentiment_idx(test_data, test_sentiment)

    # test data: pitch augmentation
    test_pitch_index = pitch_idx(test_data, test_pitch)

    # test data: speechrate augmentation
    test_speechrate_index = speechrate_idx(test_data, test_speechrate)

    # convert sparse to array in order to do negation appending
    tf_idf_test = tf_idf_test.toarray()
    tf_idf_test = np.append(tf_idf_test, test_negation_index, axis=1)
    tf_idf_test = np.append(tf_idf_test, test_category_index, axis=1)
    tf_idf_test = np.append(tf_idf_test, test_interrogation_index, axis=1)
    tf_idf_test = np.append(tf_idf_test, test_sentiment_index, axis=1)
    tf_idf_test = np.append(tf_idf_test, test_pitch_index, axis=1)
    tf_idf_test = np.append(tf_idf_test, test_speechrate_index, axis=1)

    return tf_idf_train, tf_idf_test, feature_names


# extract features and assign weights
def tf_idf_features_pure(train_data, test_data, test_category, test_sentiment, test_pitch, test_speechrate):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize, stop_words=['a','an','the','I']) #stop_words=None #stop_words=text.ENGLISH_STOP_WORDS
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data)  # bag-of-word features for training data

    # testing data：feature index extraction
    test_negation_index = negation_idx(test_data, negative_words)
    test_category_index = category_idx(test_data, test_category)
    test_sentiment_index = sentiment_idx(test_data, test_sentiment)
    test_interrogative_index = interrogative_idx(test_data, interrogative_words)
    test_pitch_index = pitch_idx(test_data, test_pitch)
    test_speechrate_index = speechrate_idx(test_data, test_speechrate)

    # convert sparse to array
    tf_idf_train = tf_idf_train.toarray()

    tf_idf_test = tf_idf_vectorize.transform(test_data)
    # convert sparse to array in order to do data augmentation
    tf_idf_test = tf_idf_test.toarray()


    return tf_idf_train, tf_idf_test, test_negation_index, test_category_index, \
            test_sentiment_index, test_interrogative_index, test_pitch_index, test_speechrate_index


def features_concatenate(train_data, test_data, train_category, test_category,
                         train_sentiment, test_sentiment, train_pitch, test_pitch,
                         train_speechrate, test_speechrate):

    # train data: get feature index
    train_negation_index = negation_idx(train_data, negative_words)
    train_category_index = category_idx(train_data, train_category)
    train_sentiment_index = sentiment_idx(train_data, train_sentiment)
    train_interrogative_index = interrogative_idx(train_data, interrogative_words)
    train_pitch_index = pitch_idx(train_data, train_pitch)
    train_speechrate_index = speechrate_idx(train_data, train_speechrate)

    # construct training features
    train_features = np.append(train_negation_index, train_category_index, axis=1)
    train_features = np.append(train_features, train_sentiment_index , axis=1)
    train_features = np.append(train_features, train_interrogative_index, axis=1)
    train_features = np.append(train_features, train_pitch_index, axis=1)
    train_features = np.append(train_features, train_speechrate_index, axis=1)

    # testing data：get feature index
    test_negation_index = negation_idx(test_data, negative_words)
    test_category_index = category_idx(test_data, test_category)
    test_sentiment_index = sentiment_idx(test_data, test_sentiment)
    test_interrogative_index = interrogative_idx(test_data, interrogative_words)
    test_pitch_index = pitch_idx(test_data, test_pitch)
    test_speechrate_index = speechrate_idx(test_data, test_speechrate)

    # construct test features
    test_features = np.append(test_negation_index, test_category_index, axis=1)
    test_features = np.append(test_features, test_sentiment_index, axis=1)
    test_features = np.append(test_features, test_interrogative_index, axis=1)
    test_features = np.append(test_features, test_pitch_index, axis=1)
    test_features = np.append(test_features, test_speechrate_index, axis=1)

    return train_features, test_features


def scott_pi(array1, array2):
    if array1.shape[0] != array2.shape[0]:
        print("ERROR!!")
        return
    agree_cnt = 0
    pos_cnt = 1
    neg_cnt = 0
    label_cnt = {pos_cnt: 0, neg_cnt: 0}

    for i in range(array1.shape[0]):
        if array1[i] == array2[i]:
            agree_cnt += 2
        label_cnt[array1[i]] += 1
        label_cnt[array2[i]] += 1
    A_0 = agree_cnt / (2 * array1.shape[0])

    A_e = (label_cnt[pos_cnt] ** 2 + label_cnt[neg_cnt] ** 2) / (4 * (2 * array1.shape[0]) ** 2)

    scott_Pi = (A_0 - A_e) / (1 - A_e)

    return scott_Pi

def get_all_info(leave_out_user, product, data, target):
    test_target = []
    test_data = []
    test_category = []
    test_sentiment = []
    test_pitch = []
    test_speechrate = []

    train_target = []
    train_data = []
    train_category = []
    train_sentiment = []
    train_pitch = []
    train_speechrate = []
    for user_num in range(1, users+1):
        user = "user"+str(user_num)
        if user_num == leave_out_user:
            test_target.extend(target[product][user])
            test_data.extend(data[product][user]["transcript"])
            test_category.extend(data[product][user]["category"])
            test_sentiment.extend(data[product][user]["sentiment"])
            test_pitch.extend(data[product][user]["pitch"])
            test_speechrate.extend(data[product][user]["speechrate"])
        else:
            train_target.extend(target[product][user])
            train_data.extend(data[product][user]["transcript"])
            train_category.extend(data[product][user]["category"])
            train_sentiment.extend(data[product][user]["sentiment"])
            train_pitch.extend(data[product][user]["pitch"])
            train_speechrate.extend(data[product][user]["speechrate"])
    # only convert target and transcript to array,
    # the rest of features will be converted to array when performing augmentation
    test_target = np.array(test_target)
    test_data = np.array(test_data)
    train_target = np.array(train_target)
    train_data = np.array(train_data)

    return test_target, test_data, test_category, test_sentiment, test_pitch, test_speechrate, \
           train_target, train_data, train_category, train_sentiment, train_pitch, train_speechrate





if __name__ == '__main__':
    # parse the input argument
    parser = argparse.ArgumentParser(description='input a model name, and a product name, within whose data that \
        the leave user product out evaluation is performed')
    parser.add_argument('modelName', type=str,
                        help='A required modle name argument: rf for random forest or svm for support vector machine')
    parser.add_argument('productName', type=str, help='A required product name argument: history | machine | remote | science ')


    args = parser.parse_args()
    modelName = args.modelName
    productName = args.productName
    if modelName not in ['rf', 'svm']:
        print('input model has to be one of rf or svm')
        exit(1)
    if productName not in ['history', 'machine', 'remote', 'science']:
        print('input product to leave out has to be one of history, machine, remote, science')
        exit(1)

    # data and target initialization
    dir = './user-data/user-data-userout'
    categories = ['Observation', 'Procedure', 'Explanation', 'Reading']
    data = None
    target = None
    # no previous loaded data, load and save them
    if not (os.path.isfile('./save/save-data-userout/data.txt')
            and os.path.isfile('./save/save-data-userout/target.txt')
            ):
        print("load data")
        data, target = load_data(dir)

        with open("./save/save-data-userout/data.txt", "wb") as d:  # Pickling
            pickle.dump(data, d)
            d.close()
        with open("./save/save-data-userout/target.txt", "wb") as t:  # Pickling
            pickle.dump(target, t)
            t.close()

    # data already loaded, retrieve them from ./save/save-data
    else:
        print('retrieve data')
        with open("./save/save-data-userout/data.txt", "rb") as d:  # Pickling
            data = pickle.load(d)
            d.close()
        with open("./save/save-data-userout/target.txt", "rb") as t:  # Pickling
            target = pickle.load(t)
            t.close()

    for product in data:
        for user in data[product]:
            for sentence in range(len(data[product][user]["transcript"])):
                # otherwise "coffee" and "coffee," would be treated as different tokens
                data[product][user]["transcript"][sentence] \
                    = data[product][user]["transcript"][sentence].replace(",", "")
                data[product][user]["transcript"][sentence] \
                    = data[product][user]["transcript"][sentence].replace(".", "")
                data[product][user]["transcript"][sentence] \
                    = data[product][user]["transcript"][sentence].replace("?", "")

    # ======  data loading finished  ======

    # === evaluation starts here ===
    users = 8
    user = 1
    print("\n\n=== Model selected: {} Current Prodcut data set being evaluated {} ===".format(modelName, productName))
    product = productName
    overall_acc = 0
    overall_recall = 0
    overall_precision = 0
    print('leave one user out evaluation:\n')
    for leave_out_user in range(1, users+1):
        print('---\nleave-out user ID: {:d}'.format(user))

        test_target, test_data, test_category, test_sentiment, test_pitch, test_speechrate, \
        train_target, train_data, train_category, train_sentiment, train_pitch, train_speechrate \
        = get_all_info(leave_out_user, product, data, target)

        # this is data augmentation vectorization
        tf_idf_train_augment, tf_idf_test_augment, feature_names = tf_idf_features(train_data, test_data,
                                                                   train_category, test_category,
                                                                   train_sentiment, test_sentiment,
                                                                   train_pitch, test_pitch,
                                                                   train_speechrate, test_speechrate
                                                                   )

        # this is pure data(no augmentation) vectorization for vote purpose
        tf_idf_train_pure, tf_idf_test_pure, \
        test_neagtion_info, test_category_info, \
        test_sentiment_info, test_interrogative_info, \
        test_pitch_info, test_speechrate_info = tf_idf_features_pure(train_data, test_data,
                                               test_category, test_sentiment,
                                               test_pitch, test_speechrate)

        # construct input vector by transcript plus additional features
        train_feature, test_feature = features_concatenate(train_data, test_data,
                                                       train_category, test_category,
                                                       train_sentiment, test_sentiment,
                                                       train_pitch, test_pitch,
                                                       train_speechrate, test_speechrate)

        # Model for transcript
        rf = RandomForestClassifier(n_estimators=300)
        svm = LinearSVC()
        if modelName == 'rf':
            model = rf
        elif modelName == 'svm':
            model = svm

        model.fit(tf_idf_train_augment, train_target)

        # predict test
        test_predict = model.predict(tf_idf_test_augment)

        test_accuracy = compute_accuracy(test_predict , test_target)
        test_recall = compute_recall(test_predict , test_target)
        test_precision = compute_precision(test_predict , test_target)

        if test_recall*test_precision != 0:
            test_F1Score = 2*test_recall*test_precision/(test_recall+test_precision)
        else:
            test_F1Score = 0

        print("test_accuracy={}".format(test_accuracy))
        print("test_recall={}".format(test_recall))
        print("test_precision={}".format(test_precision))
        print("test_F1Score={}".format(test_F1Score))

        overall_acc += test_accuracy
        overall_recall += test_recall
        overall_precision += test_precision

        user += 1

        # dump prediction back to Json file
        file_lists = os.listdir(dir + '/' + product + '/' + 'user'+str(leave_out_user) + '/')
        # one user could contain multiple files, but prediction outputs all of them in one array
        offset = 0
        for file in file_lists:
            if file.split('.')[-1] == 'json':

                with open(dir + '/' + product + '/' + 'user' + str(leave_out_user) + '/' + file, 'r') as f:
                    collection = JSON.load(f, object_pairs_hook=OrderedDict)
                f.close()
                with open(dir + '/' + product + '/' + 'user' + str(leave_out_user) + '/' + file, 'w') as out:
                    for i in range(len(collection)):
                        collection[i]["prediction"] = int(test_predict[offset])
                        offset += 1
                    JSON.dump(collection, out)
                out.close()

        if offset != len(test_predict):
            print("offset: {} output len: {}".format(offset, len(test_predict)))
            print("Error: the number of segments under user-{} does not match test output".format(leave_out_user))

    overall_acc = overall_acc / users
    overall_recall = overall_recall / users
    overall_precision = overall_precision / users
    overall_f1score = 2*overall_precision*overall_recall / (overall_precision+overall_recall)

    print("overall_acc={}".format(overall_acc))
    print("overall_precision={}".format(overall_precision))
    print("overall_recall={}".format(overall_recall))
    print("overall_F1Score={}".format(overall_f1score))

