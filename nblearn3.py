import sys
import re
import timeit
import pprint
import math

label_dict = {}
complete_list = {}
train_text_list = None
train_label_list = None
prior_positive = 0
prior_negative = 0
prior_truthful = 0
prior_deceptive = 0
total_feature_count = 0

separators = [" ", "!", ",", ")", "(", ".", "?", "\"", ":", ";", "-", "'", "/", "", '' , "$", "~", "#", "&", "{", "}",
            "<", ">"
            "by", "to", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
            "s", "t", "u", "v", "w", "x", "y", "z", "a", "about", "an", "are", "as", "at", "be", "by", "com",
            "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what",
            "when", "where", "who", "will", "with", "the"]

for index, arg in enumerate(sys.argv):
    if index == 1:
        with open(arg) as text_file:
            train_text_list = text_file.readlines()
        train_text_list = [x.strip() for x in train_text_list]
    elif index == 2:
        with open(arg) as label_file:
            train_label_list = label_file.readlines()
        train_label_list = [x.strip() for x in train_label_list]

# positive negative truthful deceptive
class EachStringInfo():
    total_positive = 0
    total_negative = 0
    total_truthful = 0
    total_deceptive = 0

    def __init__(self, identifierList, count):
        self.positive = 0
        self.negative = 0
        self.truthful = 0
        self.deceptive = 0
        self.positive_probability = 0
        self.negative_probability = 0
        self.truthful_probability = 0
        self.deceptive_probability = 0
        self.set_wordCount(identifierList, count)

    def set_positive(self, positiveCount):
        self.positive += positiveCount
        EachStringInfo.total_positive += positiveCount

    def set_negative(self, negativeCount):
        self.negative += negativeCount
        EachStringInfo.total_negative += negativeCount

    def set_truthful(self, truthfulCount):
        self.truthful += truthfulCount
        EachStringInfo.total_truthful += truthfulCount

    def set_deceptive(self, deceptiveCount):
        self.deceptive += deceptiveCount
        EachStringInfo.total_deceptive += deceptiveCount

    def set_wordCount(self, identifierList, count):
        identifier1 = identifierList[0]
        identifier2 = identifierList[1]
        if (identifier1 == "truthful"):
            self.set_truthful(count)
        elif (identifier1 == "deceptive"):
            self.set_deceptive(count)

        if (identifier2 == "positive"):
            self.set_positive(count)
        elif (identifier2 == "negative"):
            self.set_negative(count)

    def get_positive(self, positiveCount):
        return self.positive

    def get_negative(self, negativeCount):
        return self.negative

    def get_truthful(self, truthfulCount):
        return self.truthful

    def get_deceptive(self, deceptiveCount):
        return self.deceptive

    def display(self):
        return str(self.positive) + "," + str(self.negative) + "," + str(self.truthful) + "," + str(self.deceptive)

    def displayProbability(self):
        return str(self.positive_probability) + "," + str(self.negative_probability) + "," + str(self.truthful_probability) + "," + str(self.deceptive_probability)

    def checkEmptyValue(self):
        if(self.positive == 0 or self.negative == 0 or self.truthful == 0 or self.deceptive == 0):
            return True

    def addSmoothing(self):
        self.positive += 1
        self.negative += 1
        self.truthful += 1
        self.deceptive += 1
        EachStringInfo.total_positive += 1
        EachStringInfo.total_negative += 1
        EachStringInfo.total_truthful += 1
        EachStringInfo.total_deceptive += 1

    def calculateProbability(self):
        self.positive_probability = math.log(self.positive / EachStringInfo.total_positive)
        self.negative_probability = math.log(self.negative / EachStringInfo.total_negative)
        self.truthful_probability = math.log(self.truthful / EachStringInfo.total_truthful)
        self.deceptive_probability = math.log(self.deceptive / EachStringInfo.total_deceptive)

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

def tokenizedStringCount(currentText):
    unique_id = currentText.split(' ')[0]
    currentText = ' '.join(currentText.split()[1:])
    currentText = re.sub(r'([^\w])+(?=\s|$)', '', currentText)
    # currentText = re.sub("\d+", "", currentText)
    pattern  = re.compile(r"(\s+|\!|\,|\)$|\(|\.$|\?|\"|\:|\;|\-|\.{2,}|^\s\'|\/|\$|\~|\'|\{|\}|\<|\>)")
    # pattern  = re.compile(r"(\s+)")
    listToReturn = []
    listToReturn = pattern.split(currentText)
    listToReturn = [x.lower().lstrip('\'\*\.')  for x in listToReturn]
    listToReturn  = [x.strip() for x in listToReturn if x not in separators]
    listToReturn = list(filter(None, listToReturn))
    new_dict = {}
    for i in listToReturn:
        # i = stemming(i)
        new_dict[i] = new_dict.get(i, 0) + 1
    return unique_id, new_dict

def setPriorFeatureCount(labelList):
    identifier1 = labelList[0]
    identifier2 = labelList[1]
    global prior_positive, prior_negative, prior_truthful, prior_deceptive

    if (identifier1 == "truthful"):
        prior_truthful += 1
    elif (identifier1 == "deceptive"):
        prior_deceptive += 1

    if (identifier2 == "positive"):
        prior_positive += 1
    elif (identifier2 == "negative"):
        prior_negative += 1

def getWordCount():

    for eachTrainLabel in train_label_list:
        eachTrainLabelSplit = eachTrainLabel.split()
        label_dict[eachTrainLabelSplit[0]] = [eachTrainLabelSplit[1], eachTrainLabelSplit[2]]

    for index, eachTrainText in enumerate(train_text_list):
        tokenizedTextList = {}
        unique_id, tokenizedTextList = tokenizedStringCount(eachTrainText)
        labelList = label_dict[unique_id]
        setPriorFeatureCount(labelList)
        for eachWord, count in tokenizedTextList.items():
            if (eachWord in complete_list):
                complete_list[eachWord].set_wordCount(labelList, count)
            else:
                complete_list[eachWord] = EachStringInfo(labelList, count)

getWordCount()

def checkEmptyFeature():
    for key, value in complete_list.items():
        if(value.checkEmptyValue()):
            return True

checkEmptyValue = checkEmptyFeature()
total_classifier1_count = prior_positive + prior_negative
total_classifier2_count = prior_truthful + prior_deceptive

# print ("len : " + str(len(complete_list)))
# print ("P : " + str(EachStringInfo.total_positive))
# print ("N : " + str(EachStringInfo.total_negative))
# print ("T : " + str(EachStringInfo.total_truthful))
# print ("D : " + str(EachStringInfo.total_deceptive))
# print ("PP : " + str(prior_positive))
# print ("PN : " + str(prior_negative))
# print ("PT : " + str(prior_truthful))
# print ("PD : " + str(prior_deceptive))
# print ("Total Classifier 1: " + str(total_classifier1_count))
# print ("Total Classifier 2: " + str(total_classifier2_count))

def addSmoothing():
    for key, value in complete_list.items():
        value.addSmoothing()

def calculateProbability():
    for key, value in complete_list.items():
        value.calculateProbability()

addSmoothing()
calculateProbability()

nbmodelFile = open('nbmodel.txt', 'w')
nbmodelFile.write("prior_positive_probability:" + str(math.log(prior_positive/total_classifier1_count))  + "\n")
nbmodelFile.write("prior_negative_probability:" + str(math.log(prior_negative/total_classifier1_count))  + "\n")
nbmodelFile.write("prior_truthful_probability:" + str(math.log(prior_truthful/total_classifier2_count))  + "\n")
nbmodelFile.write("prior_deceptive_probability:" + str(math.log(prior_deceptive/total_classifier2_count))  + "\n")

for key in sorted(complete_list):
    nbmodelFile.write(key + " " + complete_list[key].display()  + " " +complete_list[key].displayProbability() + "\n")

nbmodelFile.close()
# for key, value in sorted(complete_list.items()):
#     if(len(key) > 7):
#         print (str(key) + '\t' * 2 + complete_list[key].display() + '\t' * 2 + complete_list[key].displayProbability())
#     else:
#         print (str(key) + '\t' * 3 + complete_list[key].display() + '\t' * 2 + complete_list[key].displayProbability())
# for key in sorted(complete_list):
#     print (key + " - " + complete_list[key].display())

# for key, value in complete_list.items():
#     print (key)
