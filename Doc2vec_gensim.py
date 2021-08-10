#-*-coding:utf-8-*-
import csv
import re
import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
import ssl
import numpy as np


def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos


#nltk.download('stopwords')
x_train = []
paperid2id = {}
id2paperid = {}
'''
reader = csv.reader(open("some_papers.csv", 'rt',encoding='utf-8'))
result = list(reader)
for i in range(len(result)):
    result[i].append(result[i][1])
    result[i].append(result[i][2])
result.pop(0)

'''
result = []
with open("../dataset/paper_titles_en_abs.txt", encoding="utf-8") as f:
    i = 0
    for line in f:
        line = line.strip().split('\t')

        if len(line) == 2:
            result.append(line[1])
        elif len(line) == 3:
            result.append(line[1] + '.' + line[2])
        else:
            result.append('')
        paperid2id[i] = line[0]
        id2paperid[line[0]] = i
        i += 1
    f.close()

stoplist = set(stopwords.words('english'))
result = [[word for word in (re.sub(' - _ 1 2 3 4 5 6 7 6 8 9 0 ` ~ ! ? @ # % & } : > ]','',"".join(result[i]).lower()).split()) if word not in stoplist] for i in range(len(result))]

for i, text in enumerate(result):
    l = len(text)
    if l > 0:
        text[l-1] = text[l-1].strip()
    document = gensim.models.doc2vec.TaggedDocument(text, tags=[i])
    x_train.append(document)

'''
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=5, epochs=20)
model.build_vocab(x_train)
model.train(x_train, total_examples=model.corpus_count, epochs=model.epochs)
model.save("model1.txt")
'''

with open("../dataset/paper_titles_after_process_abs.txt", "w", encoding="utf-8") as f:
    for i in range(len(x_train)):
        id = paperid2id[i]
        #res = model.infer_vector(x_train[i].words)
        #res = [str(x) for x in res]
        f.write(str(id) + "\t" + ' '.join(x_train[i].words) + '\n')
    f.close()
'''
model = gensim.models.doc2vec.Doc2Vec.load("model1.txt")

id1 = id2paperid[str(3917041)]
test = model.infer_vector(x_train[id1].words)
sims = model.docvecs.most_similar([test])
print("input", x_train[id1].words)
for item in sims:
    print(item[0], x_train[item[0]].words, item[1])


model = gensim.models.doc2vec.Doc2Vec.load("C:/Users/alienware/Desktop/model1.txt")
vector = model.infer_vector(["system", "response"])
list_of_vec = []
for i in range(1000):
    list_of_vec.append(model.docvecs[i])
test = pd.DataFrame(data = list_of_vec)
test.to_csv('C:/Users/alienware/Desktop/model_vec1.csv',index= False, encoding='utf-8')
print(vector)
'''
