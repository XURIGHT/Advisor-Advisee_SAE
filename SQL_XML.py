# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
import json
import pymysql
import xml.sax
from xml.sax.handler import feature_external_ges
import langid
import pandas as pd

'''
192.168.0.118
密码
8fH$aM4JTnQgW6AE
mysql root:Vis_2014 or vis_2014
'''


def add_str(s):
    return '\'' + s + '\''

data = {'key_id':[], 'title':[], 'mdate':[], 'author':[], 'pages':[], 'volume':[], 'journal':[], 'pub_year':[],
        'number':[], 'url':[], 'ee':[], 'publisher':[], 'cross_ref':[], 'book_title':[]}
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
sql_create = '''CREATE TABLE DBLP_XML (
         id int(32) UNSIGNED ZEROFILL NOT NULL AUTO_INCREMENT PRIMARY KEY,
         author varchar(1024) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '',
         title varchar(1024) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '',
         key_id varchar(512) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '' COMMENT 'The key in the xml file',
         mdate varchar(512) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '',
         pages varchar(512) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '' COMMENT 'Pages in the source, i.e. for example the journal',
         pub_year int NOT NULL DEFAULT 0,
         volume int NOT NULL DEFAULT 0 COMMENT 'Volume of the source where the publication was published',
         journal varchar(512) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '',
         number int NOT NULL DEFAULT 0 COMMENT 'Number of the source where the publication was published',
         url varchar(512) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '' COMMENT 'DBLP-internal URL (starting with db/...) where a web-page for that publication can be found on DBLP',
         ee varchar(512) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '' COMMENT 'external URL to the electronic edition of the publication',
         publisher varchar(512) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '',
         cross_ref varchar(512) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '',
         book_title varchar(512) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '' COMMENT 'Name of incollection'
  )'''
'''
cursor = db.cursor() # 定义cursor来执行SQL语句
cursor.execute(sql_create)
db.commit()
db.rollback()
'''

def add_str(s):
    s = s.replace('\'', "\"")
    return '\'' + s + '\''

paper_tags = ('article', 'inproceedings', 'proceedings', 'book', 'incollection', 'phdthesis', 'mastersthesis', 'www')


class MyHandler(xml.sax.ContentHandler):  # extract all authors
    def __init__(self):
        self.CurrentData = ""  # tag's name
        self.dict = {}  # save all authors. The key is an author's name, the value is his id
        self.name = ""  # the name of an author
        self.id = 1  # the ID of an author
        self.contents = []

        self.author = []
        self.year = 0
        self.number = 0
        self.volume = 0
        self.mdate = ""
        self.key = ""
        self.publisher = ""
        self.pages = ""
        self.journal = ""
        self.url = ""
        self.ee = ""
        self.cross_ref = ""
        self.book_title = ""
        self.title = ""

        self.title_contents = []
        self.cnt = 0
        self.total = 0

    def resolveEntity(self, publicID, systemID):
        print("TestHandler.resolveEntity(): %s %s" % (publicID, systemID))
        return systemID

    def startElement(self, tag, attributes):
        if tag != None and len(tag.strip()) > 0:
            self.CurrentData = tag

            if tag in paper_tags:
                self.key = str(attributes['key']) if attributes.__contains__('key') else ""
                self.mdate = str(attributes['mdate']) if attributes.__contains__('mdate') else ""

    def endElement(self, tag):
        if tag != None and len(tag.strip()) > 0:
            if tag in paper_tags:
                self.total += 1

                '''
                sql = "INSERT INTO DBLP_XML(author, title, key_id, mdate, pages, pub_year, volume, journal, number, url, ee, publisher, cross_ref, book_title) VALUES " \
                      "(%s, %s, %s, %s,%s, %s, %s, %s,%s, %s, %s, %s, %s, %s)" \
                      % (add_str(','.join(self.author)), add_str(self.title), add_str(self.key), add_str(self.mdate), add_str(self.pages), self.year, self.volume,
                         add_str(self.journal), self.number, add_str(self.url), add_str(self.ee), add_str(self.publisher), add_str(self.cross_ref), add_str(self.book_title))

                try:
                    cursor = db.cursor()  # 定义cursor来执行SQL语句
                    cursor.execute(sql)
                    db.commit()
                    self.cnt += 1
                except:
                    pass
                if self.cnt % 100 == 0:
                    print(self.cnt)
                '''

                data['key_id'].append(self.key)
                data['title'].append(self.title)
                data['mdate'].append(self.mdate)
                data['author'].append(','.join(self.author))
                data['pages'].append(self.pages)
                data['pub_year'].append(self.year)
                data['volume'].append(self.volume)
                data['journal'].append(self.journal)
                data['number'].append(self.number)
                data['url'].append(self.url)
                data['ee'].append(self.ee)
                data['publisher'].append(self.publisher)
                data['cross_ref'].append(self.cross_ref)
                data['book_title'].append(self.book_title)

                self.contents.clear()
                self.title_contents.clear()
                self.author.clear()
                self.year = 0
                self.number = 0
                self.volume = 0
                self.mdate = ""
                self.key = ""
                self.publisher = ""
                self.pages = ""
                self.journal = ""
                self.url = ""
                self.ee = ""
                self.cross_ref = ""
                self.book_title = ""
                self.title = ""

            elif self.CurrentData == 'author':
                self.author.append(self.name)
                self.contents.clear()

            elif self.CurrentData == 'title':
                self.title = self.title.strip()
                self.title_contents.clear()

    def characters(self, content):
        if content != '\n':
            if self.CurrentData == 'author':
                self.contents.append(content)
                self.name = ''.join(self.contents)
            # self.name += content.strip()

            elif self.CurrentData == "year":
                self.year = content.strip()

            elif self.CurrentData == "volume":
                self.volume = content.strip()

            elif self.CurrentData == "number":
                self.number = content.strip()

            elif self.CurrentData == "pages":
                self.pages = content.strip()

            elif self.CurrentData == "publisher":
                self.publisher = content.strip()

            elif self.CurrentData == "journal":
                self.journal = content.strip()

            elif self.CurrentData == "url":
                self.url = content.strip()

            elif self.CurrentData == "ee":
                self.ee = content.strip()

            elif self.CurrentData == "crossref":
                self.cross_ref = content.strip()

            elif self.CurrentData == "booktitle":
                self.book_title = content.strip()

            elif self.CurrentData == "title":
                self.title_contents.append(content)
                self.title = ''.join(self.title_contents)

            elif self.CurrentData == "i" or self.CurrentData == "sup" or self.CurrentData == "sub":
                self.title_contents.append(content)
                self.title = ''.join(self.title_contents)

if __name__ == "__main__":
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    parser.setFeature(feature_external_ges, True)

    handler1 = MyHandler()
    parser.setContentHandler(handler1)
    parser.setEntityResolver(handler1)
    parser.setDTDHandler(handler1)
    parser.parse('../dataset/dblp.xml')
    print(handler1.cnt, handler1.total)
    df_data = pd.DataFrame(data)
    df_data.to_csv('dblp_xml.csv')