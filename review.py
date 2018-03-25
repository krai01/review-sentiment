# Naive Bayes Classifier
# Karan Rai
# 2/26/18

import glob
import re
import math
import numpy

def main():
    # This will read in all data and split it into the training and testing data using the given split ratio
    xtrain, ytrain, xtest, ytest = read_in_data_function(0.67)
    # This will return a trained model that uses the given smoothing parameter
    trained_model = train(xtrain, ytrain, {"smoothparam" : 0.1})
    # This will return a predicted classification for all data in xtest given a trained model
    Y_pred = test(xtest, trained_model)
    # This will return the error rate between the predicted classifiers and the actual classifiers
    error_rate = evaluate(ytest , Y_pred)
    print('The error rate of my classifier is ' + str(error_rate))

def read_in_data_function(train_test_ratio):
    xtrain = {}
    xtest = {}
    ytrain = []
    ytest = []
    length_so_far_train = 0
    length_so_far_test = 0
    # iterates through every .txt data file, cleaning the data and adding it to the existing matrices
    for filename in glob.iglob('./*.txt'):
        file = open(filename, "r")
        # finds the total number of lines in the file to properly divide the data
        num_lines = sum(1 for line in file)
        file.close()
        cutoff = math.floor(num_lines*train_test_ratio)
        xtrain, ytrain, xtest, ytest = organize_data(xtrain, ytrain, xtest, ytest, filename, length_so_far_train, length_so_far_test, cutoff)
        # counts to keep track of the length of the training and testing data sets
        length_so_far_train += int (cutoff)
        length_so_far_test += int (num_lines -  cutoff)

    return xtrain, ytrain, xtest, ytest

def organize_data(xtrain, ytrain, xtest, ytest, filename, ltrain, ltest, cutoff):
    count = 0
    file = open(filename,"r")
    for line in file:
        count += 1
        sentence = list(set(re.sub('[^a-zA-Z ]+', '', line).lower().rstrip().split(' ')))
        for word in sentence:
            if count <= cutoff:
                if word in xtrain:
                    xtrain.get(word).append(1)
                else:
                    temp = []
                    for x in range(0, ltrain + count - 1):
                        temp.append(0)
                    temp.append(1)
                    xtrain[word] = temp
            else:
                if word in xtest:
                    xtest.get(word).append(1)
                else:
                    temp = []
                    for x in range(int(cutoff), ltest+ count - 1):
                        temp.append(0)
                    temp.append(1)
                    xtest[word] = temp
        if count <= cutoff:
            ytrain.append(int(line.replace('\n','').replace('\r','')[-1:]))
            for key, value in xtrain.iteritems():
                if len(value) < count + ltrain:
                    value.append(0)
        else:
            ytest.append(int(line.replace('\n','').replace('\r','')[-1:]))
            for key, value in xtest.iteritems():
                if len(value) < (count - cutoff) + ltest:
                    value.append(0)
    file.close()
    return xtrain, ytrain, xtest, ytest

def train(X_train, Y_train, train_opt):
    smoothing_factor = train_opt['smoothparam']
    num_keys = len(X_train.keys())
    trained_model = dict.fromkeys(X_train, [])
    pos_class_prob = sum(Y_train) / float(len(Y_train))
    y_length = len(Y_train)

    # for each word, calculate the pos and neg probabilities using the provided formula and add them to the trained model
    for word, val in X_train.items():
        count_pos = 0.0
        count_neg = 0.0
        for word_count, y_count in zip(val, Y_train):
            if word_count == 1 and y_count == 1:
                count_pos += 1
            if word_count == 1 and y_count == 0:
                count_neg += 1
        prob_neg = ((1-pos_class_prob) * count_neg + smoothing_factor) / (float(len(Y_train) - sum(Y_train)) + (smoothing_factor * num_keys))
        prob_pos = (pos_class_prob * count_pos + smoothing_factor) / (float(sum(Y_train)) + (smoothing_factor * num_keys))
        
        if prob_pos == 0:
            trained_model[word] = [prob_pos]
        else:
            trained_model[word] = [math.log(prob_pos, 10)]

        if prob_neg == 0:
            trained_model[word].append(prob_neg)
        else:
            trained_model[word].append(math.log(prob_neg, 10))
    return trained_model

def test(X_test, trained_model):
    y_pred = []
    y_pred_pos = [0.0] * len(X_test.items()[0][1])
    y_pred_neg = [0.0] * len(X_test.items()[0][1])
    intersection = [i for i in X_test if i in trained_model]
    for word_x, value in X_test.items():
        if word_x in intersection:
            for index, val in enumerate(value):
                if val == 1:
                    y_pred_pos[index] += trained_model[word_x][0]
                    y_pred_neg[index] += trained_model[word_x][1]

    # creates the predicted list for classes by comparing the positive and negative probabilities
    y_pred = numpy.greater(y_pred_pos, y_pred_neg).astype(int)
    return  y_pred
    
def evaluate(Y_test, Y_pred):
    accuracy = 1 - sum(numpy.equal(Y_test, Y_pred).astype(int)) / float(len(Y_test))
    return accuracy

main()

