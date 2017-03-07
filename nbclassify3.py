import sys
import re
import timeit
import pprint
import math
from collections import OrderedDict

model_data = None
test_data = None
stringInfoList = {}
labelListInfo = OrderedDict()

prior_positive_probability = 0
prior_negative_probability = 0
prior_truthful_probability = 0
prior_deceptive_probability = 0

for index, arg in enumerate(sys.argv):
    if index == 1:
        with open(arg) as test_file:
            test_data = test_file.readlines()
        test_data = [x.strip() for x in test_data]

model_file = open('nbmodel.txt', 'r')
model_data = model_file.readlines()
model_data = [x.strip() for x in model_data]

separators = [" ", "!", ",", ")", "(", ".", "?", "\"", ":", ";", "-", "'", "/", "", '' , "$", "~", "#", "&", "{", "}",
            "<", ">"
            "by", "to", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
            "s", "t", "u", "v", "w", "x", "y", "z", "an", "are", "as", "at", "be", "by", "com",
            "for", "from", "has", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
            "when", "where", "who", "will", "with", "the"]

class EachStringInfo():
    total_positive = 0
    total_negative = 0
    total_truthful = 0
    total_deceptive = 0

    def __init__(self, keyword, keywordCount, keywordProbability):
        self.keyword = keyword
        keywordCount = keywordCount.split(',')
        keywordProbability = keywordProbability.split(',')
        self.positive = int(keywordCount[0])
        self.negative = int(keywordCount[1])
        self.truthful = int(keywordCount[2])
        self.deceptive = int(keywordCount[3])
        self.positive_probability = float(keywordProbability[0])
        self.negative_probability = float(keywordProbability[1])
        self.truthful_probability = float(keywordProbability[2])
        self.deceptive_probability = float(keywordProbability[3])

    def get_positive(self, positiveCount):
        return self.positive

    def get_negative(self, negativeCount):
        return self.negative

    def get_truthful(self, truthfulCount):
        return self.truthful

    def get_deceptive(self, deceptiveCount):
        return self.deceptive

    def getPositiveProbability(self):
        return self.positive_probability

    def getNegativeProbability(self):
        return self.negative_probability

    def getTruthfulProbability(self):
        return self.truthful_probability

    def getDeceptiveProbability(self):
        return self.deceptive_probability

    def display(self):
        return str(self.positive) + ", " + str(self.negative) + ", " + str(self.truthful) + ", " + str(self.deceptive)

    def displayProbability(self):
        return str(self.positive_probability) + ", " + str(self.negative_probability) + ", " + str(self.truthful_probability) + ", " + str(self.deceptive_probability)


for index, eachWordInfo in enumerate(model_data):
    eachWordInfo = eachWordInfo.split(' ')
    if(len(eachWordInfo) == 3):
        stringInfoList[eachWordInfo[0]] = EachStringInfo(eachWordInfo[0], eachWordInfo[1], eachWordInfo[2])
    elif(len(eachWordInfo) == 1):
        eachWordInfo = eachWordInfo[0].split(':')
        if(len(eachWordInfo) == 2):
            if(eachWordInfo[0] == "prior_positive_probability"):
                prior_positive_probability = float(eachWordInfo[1])
            elif(eachWordInfo[0] == "prior_negative_probability"):
                prior_negative_probability = float(eachWordInfo[1])
            elif(eachWordInfo[0] == "prior_truthful_probability"):
                prior_truthful_probability =  float(eachWordInfo[1])
            elif(eachWordInfo[0] == "prior_deceptive_probability"):
                prior_deceptive_probability = float(eachWordInfo[1])

def stemming(word):
    if word.endswith('s'):
        word = word.replace('s', '')
    elif word.endswith('es'):
        word = word.replace('es', '')
    elif word.endswith('ed'):
        word = word.replace('ed', '')
    elif word.endswith('er'):
        word = word.replace('er', '')
    elif word.endswith('ly'):
        word = word.replace('ly', '')
    elif word.endswith('ing'):
        word = word.replace('ing', '')
    else:
        pass
    return word

class TestDataStringInfo():

    def __init__(self, identifierList, count):
        self.unique_id = None
        self.positive = None
        self.negative = None
        self.truthful = None
        self.deceptive = None

def tokenizedStringCount(currentText):
    unique_id = currentText.split(' ')[0]
    currentText = ' '.join(currentText.split()[1:])
    currentText = re.sub(r'([^\w])+(?=\s|$)', '', currentText)
    # currentText = re.sub("\d+", "", currentText)
    pattern  = re.compile(r"(\s+|\!|\,|\)$|\(|\.$|\?|\"|\:|\;|\-|\.{2,}|^\s\'|\/|\$|\~|\'|\{|\}|\<|\>)")
    # pattern  = re.compile(r"(\s+)")
    listToReturn = []
    listToReturn = pattern.split(currentText)
    listToReturn = [x.lower().lstrip('\'\*\.') for x in listToReturn]
    listToReturn  = [x.strip() for x in listToReturn if x not in separators]
    listToReturn = list(filter(None, listToReturn))
    return unique_id, listToReturn

# def calculatePositiveClassifier(value, eachStringInfo):
#     classifier_pp = (value * eachStringInfo.getPositiveProbability())
#     return classifier_pp
#
# def calculateNegativeClassifier(value, eachStringInfo):
#     # classifier_np = value
#     classifier_np = (value * eachStringInfo.getNegativeProbability())
#     return classifier_np
#
# def calculateTruthfulClassifier(value, eachStringInfo):
#     # classifier_tp = value
#     classifier_tp = (value * eachStringInfo.getTruthfulProbability())
#     return classifier_tp
#
# def calculateDeceptiveClassifier(value, eachStringInfo):
#     # classifier_dp = value
#     classifier_dp = (value * eachStringInfo.getDeceptiveProbability()
#     return classifier_dp

def iterateTextFile():
    for index, eachTextData in enumerate(test_data):
        tokenizedTextList = {}
        unique_id, tokenizedTextList = tokenizedStringCount(eachTextData)
        classifier_pp = 0
        classifier_np = 0
        classifier_tp = 0
        classifier_dp = 0
        # for key, value in tokenizedTextList.items():
        #     tokenizedTextList[key] = math.log(value);

        # for key, value in tokenizedTextList.items():
        #     if(key in stringInfoList):
        #         eachStringInfo = stringInfoList[key]
        #         classifier_pp += eachStringInfo.getPositiveProbability()
        #         classifier_np += eachStringInfo.getNegativeProbability()
        #         classifier_tp += eachStringInfo.getTruthfulProbability()
        #         classifier_dp += eachStringInfo.getDeceptiveProbability()
                # classifier_pp += calculatePositiveClassifier(value, eachStringInfo)
                # classifier_np += calculateNegativeClassifier(value, eachStringInfo)
                # classifier_tp += calculateTruthfulClassifier(value, eachStringInfo)
                # classifier_dp += calculateDeceptiveClassifier(value, eachStringInfo)
        for value in tokenizedTextList:
            if(value in stringInfoList):
                eachStringInfo = stringInfoList[value]
                classifier_pp += eachStringInfo.getPositiveProbability()
                classifier_np += eachStringInfo.getNegativeProbability()
                classifier_tp += eachStringInfo.getTruthfulProbability()
                classifier_dp += eachStringInfo.getDeceptiveProbability()
        classifier_pp += prior_positive_probability
        classifier_np += prior_negative_probability
        classifier_tp += prior_truthful_probability
        classifier_dp += prior_deceptive_probability
        if(classifier_tp > classifier_dp):
            labelListInfo[unique_id] = ["truthful"]
        else:
            labelListInfo[unique_id] = ["deceptive"]

        if(classifier_pp > classifier_np):
            labelListInfo[unique_id] += ["positive"]
        else:
            labelListInfo[unique_id] += ["negative"]

iterateTextFile()

f = open('nboutput.txt', 'w')

for key, value in labelListInfo.items():
    f.write(key + " " + labelListInfo[key][0]  + " " +labelListInfo[key][1] +  "\n")
#
# compare_file = open('compare.txt', 'r')
# compareData = compare_file.readlines()
# compareData = [x.strip() for x in compareData]
# compareList = {}
#
# true_deceptive = 0
# truth_deceptive = 0
# decep_truthful = 0
# true_truthful = 0
#
# true_postitive = 0
# true_negative = 0
# pos_negative = 0
# neg_positive = 0
#
# for index, eachCompareData in enumerate(compareData):
#     eachCompareDataSplit = eachCompareData.split(' ')
#     unique_id = eachCompareDataSplit[0]
#     eachClassiferInfo = labelListInfo[unique_id]
#     if(eachCompareDataSplit[1] == "deceptive" and eachClassiferInfo[0] == "deceptive"):
#         true_deceptive += 1
#     elif(eachCompareDataSplit[1] == "truthful" and eachClassiferInfo[0] == "truthful"):
#         true_truthful += 1
#     elif(eachCompareDataSplit[1] == "deceptive" and eachClassiferInfo[0] == "truthful"):
#         decep_truthful += 1
#     elif(eachCompareDataSplit[1] == "truthful" and eachClassiferInfo[0] == "deceptive"):
#         truth_deceptive += 1
#
#     if(eachCompareDataSplit[2] == "positive" and eachClassiferInfo[1] == "positive"):
#         true_postitive += 1
#     elif(eachCompareDataSplit[2] == "negative" and eachClassiferInfo[1] == "negative"):
#         true_negative += 1
#     elif(eachCompareDataSplit[2] == "positive" and eachClassiferInfo[1] == "negative"):
#         pos_negative += 1
#     elif(eachCompareDataSplit[2] == "negative" and eachClassiferInfo[1] == "positive"):
#         neg_positive += 1
#
#
# recall_positive = true_postitive / (true_postitive + pos_negative)
# recall_negative = true_negative / (true_negative + neg_positive)
# recall_truthful = true_truthful / (true_truthful + truth_deceptive)
# recall_deceptive = true_deceptive / (true_deceptive + decep_truthful)
#
# precision_positive = true_postitive / (true_postitive + neg_positive)
# precision_negative = true_negative / (true_negative + pos_negative)
# precision_truthful = true_truthful / (true_truthful + decep_truthful)
# precision_deceptive = true_deceptive / (true_deceptive + truth_deceptive)
#
# f1_positive = 2 * precision_positive * recall_positive / (precision_positive + recall_positive)
# f1_negative = 2 * precision_negative * recall_negative / (precision_negative + recall_negative)
# f1_truthful = 2 * precision_truthful * recall_truthful / (precision_truthful + recall_truthful)
# f1_deceptive = 2 * precision_deceptive * recall_deceptive / (precision_deceptive + recall_deceptive)
# avg = (f1_positive + f1_negative + f1_truthful + f1_deceptive) / 4
# print ("precision, recall, F1")
# print ("positive : " + str([precision_positive, recall_positive, f1_positive]))
# print ("negative : " + str([precision_negative, recall_negative, f1_negative]))
# print ("truthful : " + str([precision_truthful, recall_truthful, f1_truthful]))
# print ("deceptive : " + str([precision_deceptive, recall_deceptive, f1_deceptive]))
# print ("avg : " + str(avg))
