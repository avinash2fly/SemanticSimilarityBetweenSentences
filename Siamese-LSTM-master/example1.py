from lstm import *
sls=lstm("bestsem.p",load=True,training=True)

'''
test=pickle.load(open("semtest.p",'rb'), encoding='latin1')
#Example
sa="A truly wise man"
sb="He is smart"
print(sls.predict_similarity(sa,sb)*4.0+1.0)
'''
