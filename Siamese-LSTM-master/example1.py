from lstm import *
import matplotlib.pyplot as plt
import csv
from 
sls=lstm("bestsem.p",load=True,training=False)

'''
test=pickle.load(open("semtest.p",'rb'), encoding='latin1')
#Example
sa="A truly wise man"
sb="He is smart"
print(sls.predict_similarity(sa,sb)*4.0+1.0)
'''

filename = 'sample_500.csv'
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
    sc = sls.predict_similarity(x[0], x[1])*4.0+1.0
    score.append(sc[-1])


'''
index = [i for i in range(len(score))]
plt.plot(index, [score[i]*int(Y[i]==1) for i in range(len(index))], 'bx')
plt.plot(index, [score[i]*int(Y[i]==0) for i in range(len(index))], 'ro')
plt.show()
'''
