#-*-coding:utf-8-*-
import csv
import re
import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
import numpy as np
import math
import ssl
import random
import collections

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos
#�����

pairlist = [4075, 4989, 12307, 42697, 66267, 66959, 96959, 129480, 129483, 130032, 165935, 178771, 180055, 191505, 200612, 206402,239286,255361,283651,283653,288181,297935,314296,317229]

ID_pairs = []
random_ID = 0
#for i in range(len(pairlist)):
#    for j in range(i+1,len(pairlist)):
#        random_ID = math.floor(random.uniform(0,320000))
#        while random_ID==pairlist[i] or random_ID == pairlist[j]:
#            random_ID = math.floor(random.uniform(0, 320000))
#        ID_pairs.append([pairlist[i],pairlist[j],random_ID])

k=0
for i in range(math.floor(len(pairlist)/2)):
    k = 2*i
    j = 2*i+1
    random_ID = math.floor(random.uniform(0, 320000))
    while random_ID == pairlist[i] or random_ID == pairlist[j]:
        random_ID = math.floor(random.uniform(0, 320000))
    ID_pairs.append([pairlist[k],pairlist[j],random_ID])
nltk.download('stopwords')
x_train = []

reader = csv.reader(open("some_papers.csv", 'rt', encoding='utf-8'))
result = list(reader)
for i in range(len(result)):
    result[i].append(result[i][1])
    result[i].append(result[i][2])
result.pop(0)
stoplist = set(stopwords.words('english'))
result = [[word for word in (re.sub(' - _ 1 2 3 4 5 6 7 6 8 9 0 ` ~ ! @ # % & } : > ]','',"".join(result[i]).lower()).split()) if word not in stoplist] for i in range(len(result))]

for i,text in enumerate(result):
    l = len(text)
    text[l-1] = text[l-1].strip()
    document = gensim.models.doc2vec.TaggedDocument(text,tags=[i])
    x_train.append(document)

doc_id = 21000
model = gensim.models.doc2vec.Doc2Vec.load("model1.txt")
test = model.infer_vector(x_train[doc_id].words)
sims = model.docvecs.most_similar([test], topn=len(model.docvecs))
sims_csv = pd.DataFrame(data = sims)
sims_csv.to_csv('ranks.csv', encoding='utf-8')
print('Document ({}): ?{}?\n'.format(doc_id, ' '.join(x_train[doc_id].words)))
#print('Document ({}): ?{"machine", "learning","artificial","intelligence"}?\n')
#print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND', 1), ('THIRD', 2), ('Fourth', 3), ('LEAST', len(sims) - 1)]:
    print(u'%s %s: ?%s?\n' % (label, sims[index], ' '.join(x_train[sims[index][0]].words)))
#for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#    print(u'%s %s: ?%s?\n' % (label, sims[index], ' '.join(x_train[sims[index][0]].words)))
ranks = []
second_ranks = []
for doc_id in range(100, 101):
    print(doc_id)
    inferred_vector = model.infer_vector(x_train[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

count0 = 0
count1 = 0
count = 0
for x,y,z in ID_pairs:
    count0 = cos_sim(model.docvecs[x-1],model.docvecs[y-1])
    count1 = cos_sim(model.docvecs[y-1],model.docvecs[z-1])
    if abs(count0)>abs(count1):
        count+=1;
#    print('pair %s %s : %s\n' % (x-1,y-1,count0))
#    print('pair %s %s : %s\n' % (y-1,z-1,count1))
print(count/len(ID_pairs))
#counter = collections.Counter(ranks)
#print(counter)
#    second_ranks.append(sims[1])

#rank_csv = pd.DataFrame(data = ranks)
#sims_csv = pd.DataFrame(data = second_ranks)
#rank_csv.to_csv('C:/Users/alienware/Desktop/Project_Crimson/ranks.csv',index= False, encoding='utf-8')
#sims_csv.to_csv('C:/Users/alienware/Desktop/Project_Crimson/sims.csv',index= False, encoding='utf-8')


#print('Document ({}): ?{}?\n'.format(doc_id, ' '.join(x_train[doc_id].words)))
#print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
#for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#    print(u'%s %s: ?%s?\n' % (label, sims[index], ' '.join(x_train[sims[index][0]].words)))

#list_of_vec = []
#for i in range(1000):
#    list_of_vec.append(model.docvecs[i])
#test = pd.DataFrame(data = list_of_vec)
#test.to_csv('C:/Users/alienware/Desktop/model_vec1.csv',index= False, encoding='utf-8')
#print(vector)