#-*-coding:utf-8-*-
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re


def judge(word):
    flag = 0
    for char in word:
        if 'a' <= char <= 'z':
            flag = 1
        elif '0' <= char <= '9' or char == '-':
            pass
        else:
            return False
    return flag == 1


def process(word):
    ans = ""
    for char in word:
        if char.isalpha():
            ans += char
    return ans


if __name__ == "__main__":
    ids = []
    sr = stopwords.words('english')
    length = 30000
    cnt = 0
    now_index = 0
    with open("../dataset/paper_titles_en.txt", encoding="utf-8") as f:
        file_name = "../dataset/seanmf_data_" + str(now_index) + ".txt"
        ff = open(file_name, "w", encoding="utf-8")
        for line in f:
            content = line.strip('\n').split("\t")
            ids.append(content[0])
            words = word_tokenize(content[1])
            clean_words = []
            for word in words:

                word = word.lower()
                if '-' in word:
                    word = ''.join(word.split('-'))
                word = ''.join(c for c in word if c.isalpha())

                if word not in sr and judge(word):
                    if len(word) > 1:
                        clean_words.append(word)
            ff.write(' '.join(clean_words) + '\n')
            cnt += 1
            if cnt >= length:
                cnt = 0
                ff.close()
                now_index += 1
                file_name = "../dataset/seanmf_data_" + str(now_index) + ".txt"
                ff = open(file_name, "w", encoding="utf-8")

        ff.close()
        f.close()