# -*-coding:utf-8-*-
import json
import re

class Paper:
    def __init__(self, index, title, abs):
        self.index = index
        self.title = title
        self.abs = abs

title2index = {}
papers = []
index = 0
cnt = 0

with open("../dataset/paper_titles_en.txt", encoding="utf-8") as f:
    for line in f:
        content = line.strip().split("\t")
        paper = Paper(content[0], content[1], "")
        papers.append(paper)
        title2index[content[1]] = index
        index += 1
    f.close()

for i in range(4):
    file_name = "dblp-ref-" + str(i) + ".json"
    with open("../dataset/dblp.v10/dblp-ref/" + file_name, encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                title = j['title'].lower().strip('.')
                if title2index.__contains__(title):
                    t = title2index[title]
                    papers[t].abs = j['abstract'].replace('\n', '')
                    cnt += 1
            except:
                pass
        f.close()

print(cnt / index)
with open("../dataset/paper_titles_en_abs.txt", "w", encoding="utf-8") as f:
    for paper in papers:
        f.write(paper.index + '\t' + paper.title + '\t' + paper.abs + '\n')
    f.close()