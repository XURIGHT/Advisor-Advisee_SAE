# -*-coding:utf-8-*-
import json
import pymysql
import pandas as pd

data = {'key_id':[], 'title':[], 'abstract':[], 'authors':[], 'references_id':[], 'n_citation':[], 'venue':[], 'pub_year':[]}

'''
192.168.0.118
密码
8fH$aM4JTnQgW6AE
mysql root:Vis_2014 or vis_2014
'''


def add_str(s):
    return '\'' + s + '\''

'''
db = pymysql.connect(
    host='192.168.0.118',
    port=3306,
    user='root',
    password='Vis_2014',
    database='dblp',
    charset='utf8'
)
'''

sql_create = '''CREATE TABLE DBLP_JSON (
        id  int auto_increment primary key,
        key_id  varchar(100) NOT NULL,
        title  varchar(500) NOT NULL,
        abstract  TEXT,
        authors  varchar(1000),
        references_id varchar(2000),
        n_citation INT,
        venue varchar(1000) ,
        pub_year INT)'''

'''
cursor = db.cursor() # 定义cursor来执行SQL语句
try:
    cursor.execute(sql_create)
    db.commit()
except:
    # 出错时回滚（Rollback in case there is any error）
    db.rollback()
'''

def add_str(s):
    s = s.replace('\'', "\"")
    return '\'' + s + '\''

cnt = 0
total = 0
for i in range(4):
    file_name = "dblp-ref-" + str(i) + ".json"
    with open("../dataset/dblp.v10/dblp-ref/" + file_name, encoding="utf-8") as f:
        for line in f:
            total += 1
            j = json.loads(line)

            title = j['title'] if j.__contains__('title') else ""
            abs = j['abstract'] if j.__contains__('abstract') else ""
            authors = ','.join(j['authors']) if j.__contains__('authors') else ""
            n_citation = j['n_citation'] if j.__contains__('n_citation') else 0
            references = ','.join(j['references']) if j.__contains__('references') else ""
            venue = j['venue'] if j.__contains__('venue') else ""
            year = j['year'] if j.__contains__('year') else 0
            id = j['id'] if j.__contains__('id') else ""

            '''
            sql = "INSERT INTO DBLP_JSON(key_id, title, abstract, authors, references_id, n_citation, venue, pub_year) VALUES " \
                      "(%s, %s, %s, %s,%s, %s, %s, %s)" % (add_str(id), add_str(title), add_str(abs), add_str(authors),
                                                                                              add_str(references), n_citation, add_str(venue), year)

            try:
                cursor = db.cursor()  # 定义cursor来执行SQL语句
                cursor.execute(sql)
                db.commit()
                cnt += 1
            except:
                pass
            '''
            data['key_id'].append(id)
            data['title'].append(title)
            data['abstract'].append(abs)
            data['authors'].append(authors)
            data['references_id'].append(references)
            data['n_citation'].append(n_citation)
            data['venue'].append(venue)
            data['pub_year'].append(year)
            #if cnt % 100 == 0:
                #print(cnt)
        f.close()

#db.close()
print(total)

df_data = pd.DataFrame(data)
df_data.to_csv('dblp_json.csv')