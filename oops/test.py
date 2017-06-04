from lstm import *
import matplotlib.pyplot as plt
import csv
#import pickle

sls=lstm("bestsem.p",load=True,training=False)

filename = 'data.csv'
csvfile = open(filename,encoding='utf-8')
reader = csv.reader(csvfile)
reader.__next__()  # Ignoring the header
X_raw = []
Y = []
for row in reader:
    X_raw.append([row[3], row[4]])
    Y.append(int(row[5]))

score = []
for x in X_raw:
    sc = sls.predict_similarity(x[0], x[1])
    score.append(sc[-1])

#with open("score.list", "wb") as file:
#    pickle.dump(score, file)

f1_score = []
for threshold in range(100):
    countTP = 0
    countTN = 0
    countFP = 0
    countFN = 0
    for i in range(404290):
        if a[i] > threshold/100:
            if Y[i]:
                countTP += 1
            else:
                countFP += 1
        elif Y[i]:
            countFN += 1
        else:
            countTN += 1
    if countTP + countFP > 0:
        precision = countTP/(countTP + countFP)
    else:
        precision = 0
    if countTP + countFN > 0:
        recall = countTP/(countTP + countFN)
    else:
        recall = 0
    if precision + recall == 0:
        f1_score.append((threshold/100, 0, (countTP + countTN)/404190))
    else:
        f1_score.append((threshold/100, 2*precision*recall/(precision + recall), (countTP + countTN)/404190))

for i in range(100):
    print(f1_score[i])
