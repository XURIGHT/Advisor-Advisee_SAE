import xml.sax
from xml.sax.handler import feature_external_ges
import langid

paper_tags = ('article', 'inproceedings', 'proceedings', 'book', 'incollection', 'phdthesis', 'mastersthesis', 'www')


class MyHandler(xml.sax.ContentHandler):  # extract all authors
    def __init__(self):
        self.CurrentData = ""  # tag's name
        self.dict = {}  # save all authors. The key is an author's name, the value is his id
        self.name = ""  # the name of an author
        self.id = 1  # the ID of an author
        self.contents = []
        self.author = []  # all authors for the same paper
        self.year = ""  # the year of publication

        self.title_contents = []
        self.paper_id = 1
        self.title = ""
        self.title2id = {}
        self.paper_author = {}
        self.paper_year = {}
        self.paper_title = {}
        self.cnt = 0

    def resolveEntity(self, publicID, systemID):
        print("TestHandler.resolveEntity(): %s %s" % (publicID, systemID))
        return systemID

    def startElement(self, tag, attributes):
        if tag != None and len(tag.strip()) > 0:
            self.CurrentData = tag

    def endElement(self, tag):
        if tag != None and len(tag.strip()) > 0:
            if tag in paper_tags:
                self.title = self.title.lower().strip('.')
                if len(self.author) > 0:
                    for authorname in self.author:
                        authorname = authorname.strip()
                        exist = self.dict.get(authorname, -1)
                        if exist == -1:  # if this author have not been added into dict
                            self.dict[authorname] = self.id
                            self.id = self.id + 1

                exist = self.title2id.get(self.title, -1)
                if exist == -1:
                    self.title2id[self.title] = self.paper_id
                    self.paper_id += 1

                    paper_id = self.title2id[self.title]

                    self.paper_year[paper_id] = self.year
                    self.paper_title[paper_id] = self.title
                    self.paper_author[paper_id] = []
                    for authorname in self.author:
                        authorname = authorname.strip()
                        self.paper_author[paper_id].append(self.dict[authorname])
                else:
                    paper_id = self.title2id[self.title]
                    for authorname in self.author:
                        authorname = authorname.strip()
                        if self.dict[authorname] not in self.paper_author[paper_id]:
                            self.paper_year[paper_id] = ""
                            self.paper_title[paper_id] = ""
                            self.paper_author[paper_id] = []
                            break

                self.author.clear()
                self.contents.clear()
                self.title_contents.clear()

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

            elif self.CurrentData == "title":
                self.title_contents.append(content)
                self.title = ''.join(self.title_contents)

            elif self.CurrentData == "i" or self.CurrentData == "sup" or self.CurrentData == "sub":
                self.title_contents.append(content)
                self.title = ''.join(self.title_contents)


def sorted_dict(container, keys, reverse):
     aux = [(container[k], k) for k in keys]
     aux.sort(reverse=reverse)
     return aux


if __name__ == "__main__":
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    parser.setFeature(feature_external_ges, True)

    handler1 = MyHandler()
    parser.setContentHandler(handler1)
    parser.setEntityResolver(handler1)
    parser.setDTDHandler(handler1)
    parser.parse('../dataset/dblp.xml')
