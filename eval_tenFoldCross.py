from __future__ import division
import os
import json as JSON
import pickle
import random
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
from pandas import *
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import copy
import argparse


warnings.filterwarnings("ignore")


def load_label(dir):
    print("load label: {}".format(dir))
    label = {'digital-website': [], 'physical-device': []}

    for end in ['digital-website', 'physical-device']:
        file_lists = os.listdir(dir + '/' + end + '/')
        for file in file_lists:
            if file.split('.')[-1] == 'json':
                print(dir + '/' + end + '/' + file)

                with open(dir + '/' + end + '/' + file, 'r') as f:
                    collection = JSON.load(f)
                    for elem in collection:
                        label[end].append(elem['problem'])
                    f.close()

    return label


# load raw data from the current directory
def load_data(dir):
    cnt = 0
    data = {'digital-website': [], 'physical-device': []}
    target = {'digital-website': [], 'physical-device': []}
    category = {'digital-website': [], 'physical-device': []}
    sentiment = {'digital-website': [], 'physical-device': []}

    # load pitch data for each sentence for each file
    pitch = {'digital-website': [], 'physical-device': []}

    # load speech rate data for each sentence for each file
    speechrate = {'digital-website': [], 'physical-device': []}

    for end in ['digital-website', 'physical-device']:
        file_lists = os.listdir(dir+'/'+end+'/')
        for file in file_lists:
            if file.split('.')[-1] == 'json':
                print(dir+'/'+end+'/'+file)

                with open(dir+'/'+end+'/'+file, 'r') as f:
                    collection = JSON.load(f)
                    cnt += len(collection)
                    pitch_collection_perFile = []
                    speechrate_collection_perFile = []
                    for elem in collection:
                        data[end].append(elem['transcription'])
                        # idx = categories.index(elem['category'])
                        idx = elem['problem']
                        target[end].append(idx)

                        category_code = category_map[elem['category']]
                        category[end].append(category_code)

                        sentiment[end].append(elem['sentiment_gt'])

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
                        pitch_code = [0, 0]        # [if_high_pitch, if_low_pitch]
                        if abnormal_cnt.count(1) >= pitch_high_threshold:
                            pitch_code[0] = 1
                        if abnormal_cnt.count(-1) >= pitch_low_threshold:
                            pitch_code[1] = 1
                        pitch[end].append(pitch_code)
                        collection[i]['abnormal_pitch'] = pitch[end][-1]

                        # abnormal speech rate
                        seg_speechrate = collection[i]['speechrate']
                        # [if high_speechrate, if_low_speechrate]
                        speechrate_code = [0, 0]
                        if (seg_speechrate - avg_speechrate) >= 2 * std_speechrate:
                            speechrate_code[0] = 1
                        if (seg_speechrate - avg_speechrate) <= -2 * std_speechrate:
                            speechrate_code[1] = 1
                        speechrate[end].append(speechrate_code)
                        collection[i]['abnormal_speechrate'] = speechrate[end][-1]

                    f.close()

                # write abnormal result back to original json data
                with open(dir + '/' + end + '/' + file, 'w') as f:
                    JSON.dump(collection, f)
                    f.close()

    print("{0} sentences in total".format(cnt))
    return data, target, category, sentiment, pitch, speechrate


tokenize = lambda doc: doc.lower().split(" ")


# display the topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {0}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


# display the most relevant sentences within each topic
def display_sentence(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic {0}:".format(topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort(W[:,topic_idx])[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print('- '+documents[doc_index])


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
    print("test instance: {}, indeed prob: {}, detect: {}".format(len(target), total, cnt))
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
    print("test instance: {} predict as prob: {}, indeed a prob: {}".format(len(target), total, cnt))

    # no problem in ground truth targets
    if total == 0:
        return 0
    return cnt/total


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


# extract features and assign weights
def tf_idf_features(train_data, test_data, train_category, test_category, train_sentiment, test_sentiment,
                    train_pitch, test_pitch, train_speechrate, test_speechrate):
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

    # training data: sentiment augmentation
    train_sentiment_index = sentiment_idx(train_data, train_sentiment)

    # training data: interrogation augmentation
    train_interrogation_index = interrogative_idx(train_data, interrogative_words)

    # training data: pitch augmentation
    train_pitch_index = pitch_idx(train_data, train_pitch)

    # training data: speechrate augmentation
    train_speechrate_index = speechrate_idx(train_data, train_speechrate)

    # convert sparse to array in order to do negation appending
    tf_idf_train = tf_idf_train.toarray()
    tf_idf_train = np.append(tf_idf_train, train_negation_index, axis=1)
    tf_idf_train = np.append(tf_idf_train, train_category_index, axis=1)
    tf_idf_train = np.append(tf_idf_train, train_sentiment_index, axis=1)
    tf_idf_train = np.append(tf_idf_train, train_interrogation_index, axis=1)
    tf_idf_train = np.append(tf_idf_train, train_pitch_index, axis=1)
    tf_idf_train = np.append(tf_idf_train, train_speechrate_index, axis=1)

    feature_names = tf_idf_vectorize.get_feature_names()  # converts feature index to the word it represents.

    tf_idf_test = tf_idf_vectorize.transform(test_data)

    # test data: negation augmentation
    test_negation_index = negation_idx(test_data, negative_words)

    # test data: category augmentation
    test_category_index = category_idx(test_data, test_category)

    # test data: sentiment augmentation
    test_sentiment_index = sentiment_idx(test_data, test_sentiment)

    # test data: interrogation augmentation
    test_interrogation_index = interrogative_idx(test_data, interrogative_words)

    # test data: pitch augmentation
    test_pitch_index = pitch_idx(test_data, test_pitch)

    # test data: speechrate augmentation
    test_speechrate_index = speechrate_idx(test_data, test_speechrate)

    # convert sparse to array in order to do negation appending
    tf_idf_test = tf_idf_test.toarray()
    tf_idf_test = np.append(tf_idf_test, test_negation_index, axis=1)
    tf_idf_test = np.append(tf_idf_test, test_category_index, axis=1)
    tf_idf_test = np.append(tf_idf_test, test_sentiment_index, axis=1)
    tf_idf_test = np.append(tf_idf_test, test_interrogation_index, axis=1)
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

    # transcript vectorization
    tf_idf_vectorize = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True,
                                       tokenizer=tokenize, stop_words=['a', 'an', 'the',
                                                                       'I'])  # stop_words=None #stop_words=text.ENGLISH_STOP_WORDS
    train_vector = tf_idf_vectorize.fit_transform(train_data)
    train_vector = train_vector.toarray()

    test_vector = tf_idf_vectorize.transform(test_data)
    test_vector = test_vector.toarray()

    # train data: get feature index
    train_negation_index = negation_idx(train_data, negative_words)
    train_category_index = category_idx(train_data, train_category)
    train_sentiment_index = sentiment_idx(train_data, train_sentiment)
    train_interrogative_index = interrogative_idx(train_data, interrogative_words)
    train_pitch_index = pitch_idx(train_data, train_pitch)
    train_speechrate_index = speechrate_idx(train_data, train_speechrate)

    # construct training features
    train_features = np.append(train_vector, train_negation_index, axis=1)
    train_features = np.append(train_features, train_category_index, axis=1)
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
    test_features = np.append(test_vector, test_negation_index, axis=1)
    # test_features = np.append(test_features, test_category_index, axis=1)
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

    # print("A_0 = {}  A_e = {}".format(A_0, A_e))
    scott_Pi = (A_0 - A_e) / (1 - A_e)

    return scott_Pi


category_map = {'Observation': [0, 0], 'Procedure': [0, 1], 'Explanation': [1, 0], 'Reading': [1, 1]}

negative_words = ['no', 'not', 'don\'t', 'doesn\'t', 'didn\'t', 'never',
                      'but', 'nope', 'wasn\'t', 'won\'t', 'aren\'t', 'weren\'t',
                      'none', 'haven\'t', 'ain\'t', 'wouldn\'t']

interrogative_words = ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how',
                        'What', 'When', 'Where', 'Which', 'Who', 'Whom', 'Whose', 'Why', 'How']


if __name__ == '__main__':
    # parse the input argument
    parser = argparse.ArgumentParser(description='input a model name')
    parser.add_argument('modelName', type=str,
                        help='A required modle name argument: rf for random forest or svm for support vector machine')
    args = parser.parse_args()
    model = args.modelName
    if model not in ['rf', 'svm']:
        print('input model has to be one of rf or svm')
        exit(1)

    # data and target initialization
    dir = './user-data/user-data-all-fields'
    categories = ['Observation', 'Procedure', 'Explanation', 'Reading']
    data = None
    target = None
    # no previous loaded data, load and save them
    if not (os.path.isfile('./save/save-data/digital-website/data.txt')
            and os.path.isfile('./save/save-data/digital-website/target.txt')
            and os.path.isfile('./save/save-data/digital-website/category.txt')
            and os.path.isfile('./save/save-data/physical-device/data.txt')
            and os.path.isfile('./save/save-data/physical-device/target.txt')
            and os.path.isfile('./save/save-data/physical-device/category.txt')
            and os.path.isfile('./save/save-data/physical-device/sentiment.txt')
            and os.path.isfile('./save/save-data/digital-website/sentiment.txt')
            and os.path.isfile('./save/save-data/physical-device/pitch.txt')
            and os.path.isfile('./save/save-data/digital-website/pitch.txt')
            and os.path.isfile('./save/save-data/physical-device/speechrate.txt')
            and os.path.isfile('./save/save-data/digital-website/speechrate.txt')
            ):
        print("load data")
        data, target, category, sentiment, pitch, speechrate = load_data(dir)

        website_pitch = pitch['digital-website']
        website_speechrate = speechrate['digital-website']
        website_sentiment = sentiment['digital-website']
        website_category = category['digital-website']
        website_data = data['digital-website']
        website_target = target['digital-website']

        device_pitch = pitch['physical-device']
        device_speechrate = speechrate['physical-device']
        device_sentiment = sentiment['physical-device']
        device_category = category['physical-device']
        device_data = data['physical-device']
        device_target = target['physical-device']

        # digital data
        with open("./save/save-data/digital-website/data.txt", "wb") as ddf:  # Pickling
            pickle.dump(data['digital-website'], ddf, protocol=2)
            ddf.close()
        with open("./save/save-data/digital-website/target.txt", "wb") as dtf:  # Pickling
            pickle.dump(target['digital-website'], dtf, protocol=2)
            dtf.close()
        with open("./save/save-data/digital-website/category.txt", "wb") as dcf:  # Pickling
            pickle.dump(category['digital-website'], dcf, protocol=2)
            dcf.close()
        with open("./save/save-data/digital-website/sentiment.txt", "wb") as dsf:  # Pickling
            pickle.dump(sentiment['digital-website'], dsf)
            dsf.close()
        with open("./save/save-data/digital-website/pitch.txt", "wb") as dpf:  # Pickling
            pickle.dump(pitch['digital-website'], dpf)
            dpf.close()
        with open("./save/save-data/digital-website/speechrate.txt", "wb") as dsrf:  # Pickling
            pickle.dump(speechrate['digital-website'], dsrf)
            dsrf.close()

        # physical data
        with open("./save/save-data/physical-device/data.txt", "wb") as pdf:  # Pickling
            pickle.dump(data['physical-device'], pdf, protocol=2)
            pdf.close()
        with open("./save/save-data/physical-device/target.txt", "wb") as ptf:  # Pickling
            pickle.dump(target['physical-device'], ptf, protocol=2)
            ptf.close()
        with open("./save/save-data/physical-device/category.txt", "wb") as pcf:  # Pickling
            pickle.dump(category['physical-device'], pcf, protocol=2)
            pcf.close()
        with open("./save/save-data/physical-device/sentiment.txt", "wb") as psf:  # Pickling
            pickle.dump(sentiment['physical-device'], psf)
            psf.close()
        with open("./save/save-data/physical-device/pitch.txt", "wb") as ppf:  # Pickling
            pickle.dump(pitch['physical-device'], ppf)
            ppf.close()
        with open("./save/save-data/physical-device/speechrate.txt", "wb") as psrf:  # Pickling
            pickle.dump(speechrate['physical-device'], psrf)
            psrf.close()

    # data already loaded, retrieve them from ./save-data
    else:
        print('retrieve data')

        # digital data
        with open("./save/save-data/digital-website/data.txt", "rb") as ddf:  # Pickling
            website_data = pickle.load(ddf)
            ddf.close()
        with open("./save/save-data/digital-website/target.txt", "rb") as dtf:  # Pickling
            website_target = pickle.load(dtf)
            dtf.close()
        with open("./save/save-data/digital-website/category.txt", "rb") as dcf:  # Pickling
            website_category = pickle.load(dcf)
            dcf.close()
        with open("./save/save-data/digital-website/sentiment.txt", "rb") as dsf:  # Pickling
            website_sentiment = pickle.load(dsf)
            dsf.close()
        with open("./save/save-data/digital-website/pitch.txt", "rb") as dpf:  # Pickling
            website_pitch = pickle.load(dpf)
            dpf.close()
        with open("./save/save-data/digital-website/speechrate.txt", "rb") as dsrf:  # Pickling
            website_speechrate = pickle.load(dsrf)
            dsrf.close()

        # physical data
        with open("./save/save-data/physical-device/data.txt", "rb") as pdf:  # Pickling
            device_data = pickle.load(pdf)
            pdf.close()
        with open("./save/save-data/physical-device/target.txt", "rb") as ptf:  # Pickling
            device_target = pickle.load(ptf)
            ptf.close()
        with open("./save/save-data/physical-device/category.txt", "rb") as pcf:  # Pickling
            device_category = pickle.load(pcf)
            ptf.close()
        with open("./save/save-data/physical-device/sentiment.txt", "rb") as psf:  # Pickling
            device_sentiment = pickle.load(psf)
            psf.close()
        with open("./save/save-data/physical-device/pitch.txt", "rb") as ppf:  # Pickling
            device_pitch = pickle.load(ppf)
            ppf.close()
        with open("./save/save-data/physical-device/speechrate.txt", "rb") as psrf:  # Pickling
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
        data[i] = data[i].replace(".", "")
        data[i] = data[i].replace("?", "")

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

    # ====== data loading finished ======
    data = np.array(data)
    category = np.array(category)
    sentiment = np.array(sentiment)
    target = np.array(target)
    pitch = np.array(pitch)
    speechrate = np.array(speechrate)

    # === 10 fold cross validation starts here ===
    modelName = None
    if model == 'rf':
        modelName = 'Random Forset'
    if model == 'svm':
        modelName = 'SVM'
    print("\n\n=== model selected: {} ===".format(modelName))
    x_fold = 10
    kf = KFold(n_splits=x_fold)
    foldID = 1
    overall_acc = 0
    overall_recall = 0
    overall_precision = 0
    print('10 fold cross validation:\n')
    for train_index, test_index in kf.split(data):
        print('\nfold ID: {:d}\n'.format(foldID))

        # this is data augmentation vectorization
        tf_idf_train, tf_idf_test, feature_names = tf_idf_features(data[train_index], data[test_index],
                                                                   category[train_index], category[test_index],
                                                                   sentiment[train_index], sentiment[test_index],
                                                                   pitch[train_index], pitch[test_index],
                                                                   speechrate[train_index], speechrate[test_index]
                                                                   )

        # train data: get feature index
        train_negation_info = negation_idx(data[train_index], negative_words)
        train_category_info = category_idx(data[train_index], category[train_index])
        train_sentiment_info = sentiment_idx(data[train_index], sentiment[train_index])
        train_interrogative_info = interrogative_idx(data[train_index], interrogative_words)
        train_pitch_info = pitch_idx(data[train_index], pitch[train_index])
        train_speechrate_info = speechrate_idx(data[train_index], speechrate[train_index])

        # this is pure data(no augmentation) vectorization for vote purpose
        tf_idf_train_pure, tf_idf_test_pure, \
        test_negation_info, test_category_info, \
        test_sentiment_info, test_interrogative_info, \
        test_pitch_info, test_speechrate_info = tf_idf_features_pure(data[train_index], data[test_index],
                                               category[test_index], sentiment[test_index],
                                               pitch[test_index], speechrate[test_index])

        # construct input vector by transcript plus additional features
        train_feature, test_feature = features_concatenate(data[train_index], data[test_index],
                                                       category[train_index], category[test_index],
                                                       sentiment[train_index], sentiment[test_index],
                                                       pitch[train_index], pitch[test_index],
                                                       speechrate[train_index], speechrate[test_index])

        train_target, test_target = target[train_index], target[test_index]

        # ========================== STRAT =========================
        # ======= automatically generate test category info ========
        category_train_y = []
        for i in range(len(category[train_index])):
            if np.array_equal(category[train_index][i], [0,0]):
                category_train_y.append(0)
            elif np.array_equal(category[train_index][i], [0,1]):
                category_train_y.append(1)
            elif np.array_equal(category[train_index][i], [1,0]):
                category_train_y.append(2)
            elif np.array_equal(category[train_index][i], [1,1]):
                category_train_y.append(3)
            else:
                print("error, 5th category occurs")
        category_train_y = np.array(category_train_y)
        svm_category = LinearSVC(C=0.5)
        svm_category.fit(tf_idf_train_pure, category_train_y)
        auto_category = svm_category.predict(tf_idf_test_pure)

        auto_category_info = []
        for i in range(len(auto_category)):
            if auto_category[i] == 0:
                auto_category_info.append([0,0])
            elif auto_category[i] == 1:
                auto_category_info.append([0,1])
            elif auto_category[i] == 2:
                auto_category_info.append([1,0])
            elif auto_category[i] == 3:
                auto_category_info.append([1,1])
            else:
                print(auto_category[i])
                print("error, 5th category occurs")
        auto_category_info = np.array(auto_category_info)

        # generate training and test data(auto category)
        tf_idf_test_autoCategory = np.append(tf_idf_test_pure, auto_category_info, axis=1)
        tf_idf_train = np.append(tf_idf_train_pure, train_category_info, axis=1)

        # ======= automatically generate test category info ========
        # ======================== END =============================

        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        svm = LinearSVC(C=0.5)
        if model == 'rf':
            model = rf
        elif model == 'svm':
            model = svm

        model.fit(tf_idf_train, train_target)

        test_predict = model.predict(tf_idf_test_autoCategory)

        cnt = 0
        for i in range(test_target.shape[0]):
            if test_predict[i] == test_target[i]:
                cnt += 1
        test_accuracy = cnt / test_target.shape[0]
        test_recall = compute_recall(test_predict, test_target)
        test_precision = compute_precision(test_predict, test_target)

        print("test_accuracy={}".format(test_accuracy))
        print("test_recall={}".format(test_recall))
        print("test_precision={}".format(test_precision))

        overall_acc += test_accuracy
        overall_recall += test_recall
        overall_precision += test_precision

        foldID += 1
    overall_acc = overall_acc / x_fold
    overall_recall = overall_recall / x_fold
    overall_precision = overall_precision / x_fold
    overall_F1Score = 2 * overall_recall * overall_precision / (overall_recall + overall_precision)

    print("overall_acc={}".format(overall_acc))
    print("overall_precision={}".format(overall_precision))
    print("overall_recall={}".format(overall_recall))
    print("overall_F1Score={}".format(overall_F1Score))

