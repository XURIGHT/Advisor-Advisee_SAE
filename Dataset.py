#-*-coding:utf-8-*-
import os
from Node import *
from Edge import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import re
import gensim


class Dataset:
    def __init__(self, path):
        self.author_id_inc = 0
        self.path = path
        self.nodes = {}
        self.edges = []
        self.paper2year = {}
        self.paper2title = {}
        self.edge_exist = {}
        self.get_edge = {}
        self.authorname2id = {}
        self.pos = {}
        self.neg = {}
        self.paper2authors = {}
        self.labeled_author_pair = {}
        self.labeled_authors_index = {}
        self.labeled_author_name = {}
        self.paperid2vec = {}
        self.min_year = 1970
        self.max_year = 2021

        self.load_title_sim_feature()
        self.load_data()
        self.filter_data()
        # self.write_file()
        # self.test_edges()
        # self.test_roc()

    def load_data(self):
        author_filename = "dblp_author.txt"
        year_filename = "dblp_paper_year_title.txt"
        paper_author_filename = "dblp_paper_author.txt"

        with open(os.path.join(self.path, "pos.txt"), encoding="utf-8") as f:
            for line in f:
                content = line.strip('\n').split("\t")
                self.pos[content[0] + '#' + content[1]] = 1
                self.labeled_author_pair[content[0] + '#' + content[1]] = 1
                self.labeled_author_name[content[0]] = 1
                self.labeled_author_name[content[1]] = 1
            f.close()

        with open(os.path.join(self.path, "neg.txt"), encoding="utf-8") as f:
            for line in f:
                content = line.strip('\n').split("\t")
                self.neg[content[0] + '#' + content[1]] = 1
                self.labeled_author_pair[content[0] + '#' + content[1]] = 1
                self.labeled_author_name[content[0]] = 1
                self.labeled_author_name[content[1]] = 1
            f.close()

        with open(os.path.join(self.path, author_filename), encoding="utf-8") as f:
            for line in f:
                content = line.strip('\n').split("\t")
                t_name = content[1][:-5] if '0' in content[1] else content[1]
                if self.labeled_author_name.__contains__(t_name):
                    node = Node(int(content[0]), content[1])
                    self.authorname2id[content[1]] = int(content[0])
                    self.nodes[int(content[0])] = node
            f.close()

        with open(os.path.join(self.path, year_filename), encoding="utf-8") as f:
            for line in f:
                content = line.strip('\n').split("\t")
                self.paper2year[int(content[0])] = int(content[2])
                self.paper2title[int(content[0])] = content[1]

            f.close()

        with open(os.path.join(self.path, paper_author_filename)) as f:
            for line in f:
                content = line.strip('\n').split("\t")
                paper_index = int(content[0])
                author_index = int(content[1])
                if self.paper2authors.__contains__(paper_index):
                    self.paper2authors[paper_index].append(author_index)
                else:
                    self.paper2authors[paper_index] = [author_index]
            f.close()

        with open(os.path.join(self.path, paper_author_filename)) as f:
            for line in f:
                content = line.strip('\n').split("\t")
                paper_index = int(content[0])
                author_index = int(content[1])
                if self.nodes.__contains__(author_index):
                    self.nodes[author_index].add_paper(self.paper2year[paper_index], paper_index, self.paper2authors[paper_index])
            f.close()

        cnt = 0
        for author_pair in self.labeled_author_pair.keys():
            authors = author_pair.split('#')
            if self.authorname2id.__contains__(authors[0]):
                self.make_edge(self.authorname2id[authors[0]])
            else:
                id = self.match(authors[0], authors[1])
                if id != -1:
                    self.make_edge(id)
                    self.authorname2id[authors[0]] = id
                else:
                    #print(0, authors[0], authors[1])
                    cnt += 1

            if self.authorname2id.__contains__(authors[1]):
                self.make_edge(self.authorname2id[authors[1]])
            else:
                id = self.match(authors[1], authors[0])
                if id != -1:
                    self.make_edge(id)
                    self.authorname2id[authors[1]] = id
                else:
                    #print(1, authors[0], authors[1])
                    cnt += 1
        print(cnt)

    def filter_data(self):
        ans_edges = []
        for edge in self.edges:
            node1 = self.nodes[edge.node1]
            node2 = self.nodes[edge.node2]
            if node2.start_year >= node1.start_year:  # Assumption 2
                continue
            if len(edge.years) <= 1:  # R3
                continue
            if node2.start_year + 2 > edge.st:  # R4
                continue

            #self.cal_factors(edge)

            #if not self.test_r1(edge):  # R1
                #continue
            #if not self.test_r2(edge):  # R2
            #    continue

            ans_edges.append(edge)

        self.edges = ans_edges
        '''
        for edge in self.edges:
            edge.set_ed(self.cal_ed(edge))
            edge.set_l(self.cal_l(edge))
            self.nodes[edge.node1].teachers.append(edge.node2)
            self.nodes[edge.node2].students.append(edge.node1)

        for node in self.nodes:
            edge = Edge(node.index, 0)
            edge.l = 1
            edge.st = 1000000000
            edge.ed = 0
            self.edges.append(edge)
            self.nodes[edge.node1].teachers.append(0)
            self.nodes[0].students.append(edge.node1)
        '''

        self.edges.sort(key=lambda x: x.node1, reverse=False)

        for index in range(len(self.edges)):
            s1 = str(self.edges[index].node1) + "#" + str(self.edges[index].node2)
            self.get_edge[s1] = index

    def node_sum(self, node, t):
        sum1 = 0
        for index in range(len(node.nyears)):
            if node.years[index] > t:
                break
            sum1 += node.nyears[index]
        return sum1

    def edge_sum(self, edge, t):
        sum1 = 0
        for index in range(len(edge.nyears)):
            if edge.years[index] > t:
                break
            sum1 += edge.nyears[index]
        return sum1

    def cal_factors(self, edge):
        node1 = self.nodes[edge.node1]
        node2 = self.nodes[edge.node2]
        for time in range(edge.years[0], edge.years[len(edge.years) - 1] + 1):
            abefore = self.node_sum(node1, time)
            bbefore = self.node_sum(node2, time)
            abbefore = self.edge_sum(edge, time)
            edge.kulcs.append(abbefore/(2.0 * abefore) + abbefore/(2.0 * bbefore))
            edge.irs.append(1.0 * (bbefore - abefore)/(abefore + bbefore - abbefore))

        for time in range(edge.years[0], edge.years[len(edge.years) - 1] + 1):
            if time in edge.years:
                abefore = abefore - node1.nyears[self.find_year(node1.years, time)] if self.find_year(node1.years, time) != -1 else abefore
                bbefore = bbefore - node2.nyears[self.find_year(node2.years, time)] if self.find_year(node2.years, time) != -1 else bbefore
                abbefore = abbefore - edge.nyears[self.find_year(edge.years, time)] if self.find_year(edge.years, time) != -1 else abbefore

            if abefore != 0 and bbefore != 0:
                edge.f3s.append(abbefore / (2.0 * abefore) + abbefore / (2.0 * bbefore))
            else:
                tmp = edge.f3s[len(edge.f3s) - 1]
                edge.f3s.append(tmp)

    def find_year(self, year_list, year):
        for index in range(len(year_list)):
            if year == year_list[index]:
                return index
        return -1

    def kulc(self, edge, t):
        node1 = self.nodes[edge.node1]
        node2 = self.nodes[edge.node2]

        return (self.edge_sum(edge, t) / 2.0) * (1.0 / self.node_sum(node1, t) + 1.0 / self.node_sum(node2, t))

    def ir(self, edge, t):
        node1 = self.nodes[edge.node1]
        node2 = self.nodes[edge.node2]

        sum_edge = self.edge_sum(edge, t)
        sum_nodei = self.node_sum(node1, t)
        sum_nodej = self.node_sum(node2, t)

        return (1.0 * (sum_nodej - sum_nodei)) / (sum_nodei + sum_nodej - sum_edge)

    def test_r1(self, edge):
        for time in range(edge.years[0], edge.years[len(edge.years) - 1] + 1):
            if self.ir(edge, time) > 0:
                return True
        return False

    def test_r2(self, edge):
        kulcs = []
        for time in range(edge.years[0], edge.years[len(edge.years) - 1] + 1):
            kulc = self.kulc(edge, time)
            if len(kulcs) == 0 or kulc <= kulcs[len(kulcs) - 1]:
                kulcs.append(kulc)
            else:
                return True
        return False

    def cal_ed(self, edge):
        year1 = edge.years[len(edge.years) - 1]
        year2 = edge.years[len(edge.years) - 1]

        mab = edge.years[0]
        m = max(self.nodes[edge.node1].years[0], self.nodes[edge.node2].years[0])

        s = []
        for index in range(len(edge.kulcs)):
            s.append(edge.kulcs[index] - edge.f3s[index])
        for index in range(len(s) - 1):
            if s[index] > s[index + 1]:
                year1 = edge.years[0] + index
                break

        delta = [edge.kulcs[index + 1] - edge.kulcs[index] for index in range(len(edge.kulcs) - 1)]
        peak = [index for index in range(len(delta)) if delta[index] > 0]
        if len(peak) != 0:
            s = edge.kulcs[:]
            for index in range(peak[0], len(s) - 1):
                if s[index] > s[index + 1]:
                    year2 = edge.years[0] + index
                    break

        return min(year1, year2)

    def cal_l(self, edge):
        sum1 = 0
        for t in range(edge.st, edge.ed + 1):
            sum1 += self.kulc(edge, t) + self.ir(edge, t)
        sum1 = 1.0 * sum1 / (2 * (edge.ed - edge.st + 1))
        return sum1

    def write_file(self):
        with open(os.path.join(self.path, "my_edge.txt"), "w") as f:
            for edge in self.edges:
                f.write(str(edge.node1) + " " + str(edge.node2) + " " + str(round(edge.l, 6)) + " " + str(edge.st) + " " + str(edge.ed) + "\n")

    def test_edges(self):
        self.right_edges = {}
        right_cnt = 0

        with open(os.path.join(self.path, "edge.txt")) as f:
            cnt = 0
            for line in f:
                if cnt == 0:
                    cnt += 1
                    continue
                content = line.split(" ")
                s = content[0] + "#" + content[1]
                self.right_edges[s] = []
                self.right_edges[s].append(float(content[2]))
                self.right_edges[s].append(int(content[3]))
                self.right_edges[s].append(int(content[4]))
                cnt += 1

            for edge in self.edges:
                s = str(edge.node1) + "#" + str(edge.node2)
                if s in self.right_edges.keys():
                    if edge.st == self.right_edges[s][1] and edge.ed == self.right_edges[s][2] and edge.l - self.right_edges[s][0] < 1e-5:
                        right_cnt += 1
            print(right_cnt / len(self.right_edges))

    def generate_sets(self):
        cnt = 2166
        new_pos = {}
        new_neg = {}
        new_pos_keys = random.sample(self.pos.keys(), cnt)
        new_neg_keys = random.sample(self.neg.keys(), cnt)

        for key in new_pos_keys:
            new_pos[key] = 1
        for key in new_neg_keys:
            new_neg[key] = 1

        return new_pos, new_neg

    def test_roc(self):
        with open(os.path.join(self.path, "pos.txt"), encoding="utf-8") as f:
            for line in f:
                content = line.strip('\n').split("\t")
                s = str(self.authorname2id[content[0]]) + '#' + str(self.authorname2id[content[1]])
                self.pos[s] = 1

        with open(os.path.join(self.path, "neg.txt"), encoding="utf-8") as f:
            for line in f:
                content = line.strip('\n').split("\t")
                s = str(self.authorname2id[content[0]]) + '#' + str(self.authorname2id[content[1]])
                self.neg[s] = 1

        res = []
        with open(os.path.join(self.path, "adv.txt")) as f:
            cnt = 0
            for line in f:
                if cnt == 0:
                    cnt += 1
                    continue
                content = line.split(' ')
                n1 = content[0]
                n2 = content[1]
                if n2 == "0":
                    continue
                p = float(content[2])
                s = n1 + "#" + n2
                res.append([s, p])

        self.print_acc(res, 2)
        self.draw_roc(res)

    def draw_roc(self, res):

        y_label = []
        y_pre = []
        for item in res:
            if item[0] in self.pos.keys():
                y_label.append(1)
                y_pre.append(item[1])
            elif item[0] in self.neg.keys():
                y_label.append(0)
                y_pre.append(item[1])

        fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=1)
        tpr[0] = tpr[1]
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, '-', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2, color='blue')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def print_acc(self, res, k):
        new_pos, new_neg = self.generate_sets()
        t = 0
        f = 0
        a = np.array([x[1] for x in res])
        cit = np.percentile(a, 75)
        index = 0
        while index < len(res):
            last_id = res[index][0].split("#")[0]
            now_id = last_id
            tmp = []
            while now_id == last_id:
                tmp.append(res[index])
                index += 1
                if index >= len(res):
                    break
                now_id = res[index][0].split("#")[0]
            tmp.sort(key=lambda x: x[1], reverse=True)

            tmp = tmp[:min(k, len(tmp))]

            for item in tmp:
                if item[1] >= cit:
                    if item[0] in new_pos.keys():
                        t += 1
                    elif item[0] in new_neg.keys():
                        f += 1
        print(t, f)
        print("P@(%d, %f) acc is %.5f" % (k, cit, 1.0 * t / (t + f)))

    def match_one_name(self, name1, name2):
        name1_parts = re.split('[-\s]', name1)
        name2_parts = re.split('[-\s]', name2)
        n1_in_n2 = 0
        n2_in_n1 = 0
        for name1_part in name1_parts:
            if name1_part in name2_parts:
                n1_in_n2 += 1
        for name2_part in name2_parts:
            if name2_part in name1_parts:
                n2_in_n1 += 1

        n1_in_n2 /= len(name1_parts)
        n2_in_n1 /= len(name2_parts)

        return max(n1_in_n2, n2_in_n1) > 0.6

    def match(self, name1, name2):
        if self.authorname2id.__contains__(name2):
            for id in self.nodes[self.authorname2id[name2]].co_authors.keys():
                if self.nodes.__contains__(id) and self.match_one_name(name1, self.nodes[id].name):
                    return id
        else:
            for i in range(1, 100):
                n1 = name1 + " " + str(i).zfill(4)
                if self.authorname2id.__contains__(n1) is False:
                    break
                else:
                    for id in self.nodes[self.authorname2id[n1]].co_authors.keys():
                        if self.nodes.__contains__(id) and self.match_one_name(name2, self.nodes[id].name):
                            return id

        return -1

    def make_edge(self, index1):
        for index2 in self.nodes[index1].co_authors.keys():
            if self.nodes.__contains__(index2):
                s1 = str(index1) + "#" + str(index2)
                s2 = str(index2) + "#" + str(index1)

                if self.edge_exist.__contains__(s1) is False:
                    self.edges.append(Edge(index1, index2))
                    self.edge_exist[s1] = len(self.edges) - 1

                    for paper_index in self.nodes[index1].co_authors[index2]:
                        if (self.nodes[index1].fir_dic.__contains__(paper_index) and self.nodes[index2].sec_dic.__contains__(paper_index)) or \
                                (self.nodes[index1].sec_dic.__contains__(paper_index) and self.nodes[index2].fir_dic.__contains__(paper_index)):
                            self.edges[self.edge_exist[s1]].add_paper(self.paper2year[paper_index], "fir_sec")

                        elif (self.nodes[index1].fir_dic.__contains__(paper_index) and self.nodes[index2].last_dic.__contains__(paper_index)) or \
                                (self.nodes[index1].last_dic.__contains__(paper_index) and self.nodes[index2].fir_dic.__contains__(paper_index)):
                            self.edges[self.edge_exist[s1]].add_paper(self.paper2year[paper_index], "fir_last")

                        else:
                            self.edges[self.edge_exist[s1]].add_paper(self.paper2year[paper_index], "")

                if self.edge_exist.__contains__(s2) is False:
                    self.edges.append(Edge(index2, index1))
                    self.edge_exist[s2] = len(self.edges) - 1

                    for paper_index in self.nodes[index2].co_authors[index1]:
                        if (self.nodes[index1].fir_dic.__contains__(paper_index) and self.nodes[index2].sec_dic.__contains__(paper_index)) or \
                                (self.nodes[index1].sec_dic.__contains__(paper_index) and self.nodes[index2].fir_dic.__contains__(paper_index)):
                            self.edges[self.edge_exist[s2]].add_paper(self.paper2year[paper_index], "fir_sec")

                        elif (self.nodes[index1].fir_dic.__contains__(paper_index) and self.nodes[index2].last_dic.__contains__(paper_index)) or \
                                (self.nodes[index1].last_dic.__contains__(paper_index) and self.nodes[index2].fir_dic.__contains__(paper_index)):
                            self.edges[self.edge_exist[s2]].add_paper(self.paper2year[paper_index], "fir_last")

                        else:
                            self.edges[self.edge_exist[s2]].add_paper(self.paper2year[paper_index], "")

    def get_features(self):
        pos_cnt = 0
        neg_cnt = 0
        both_cnt = 0
        with open('../dataset/features_matrix.txt', 'w', encoding='utf-8') as f:
            for edge in self.edges:
                ai = self.nodes[edge.node1]
                aj = self.nodes[edge.node2]
                key_str = ai.name + "#" + aj.name
                if self.labeled_author_pair.__contains__(key_str):
                    if self.pos.__contains__(key_str) and self.neg.__contains__(key_str):
                        print(key_str)
                        both_cnt += 1
                    elif self.pos.__contains__(key_str) or self.neg.__contains__(key_str):

                        if self.labeled_authors_index.__contains__(edge.node1) is False:
                            self.labeled_authors_index[edge.node1] = 1
                        if self.labeled_authors_index.__contains__(edge.node2) is False:
                            self.labeled_authors_index[edge.node2] = 1

                        #edge.sim = self.get_title_sim_feature(edge)

                        f.write(str(edge.node1) + '\t' + str(edge.node2) + '\n')
                        f.write('\t'.join([str(x) for x in edge.nyears]) + '\n')
                        f.write('\t'.join([str(x) for x in ai.nyears]) + '\n')
                        f.write('\t'.join([str(x) for x in aj.nyears]) + '\n')
                        f.write('\t'.join([str(x) for x in edge.fir_sec]) + '\n')
                        f.write('\t'.join([str(x) for x in edge.fir_last]) + '\n')

                        if self.pos.__contains__(key_str):
                            f.write("1" + '\t')  # 标签
                            pos_cnt += 1
                            if edge.node1 == 735955:
                                print("advisee:", ai.name)
                                for pl in ai.papers:
                                    for p in pl:
                                        print(p, self.paper2title[p])

                                print("advisor:", aj.name)
                                for pl in aj.papers:
                                    for p in pl:
                                        print(p, self.paper2title[p])
                        else:
                            f.write("0" + '\t')  # 标签
                            neg_cnt += 1
                        f.write('\n')
            f.close()

            print("both: ", both_cnt)
            print("total: ", pos_cnt + neg_cnt)
            print("pos: ", pos_cnt)
            print("neg: ", neg_cnt)

    def print_titles(self):
        finished_paper = {}
        cnt = 0
        with open('../dataset/paper_titles.txt', 'w', encoding='utf-8') as f:
            for author_index in self.labeled_authors_index.keys():
                node = self.nodes[author_index]

                for paper_list in node.papers:
                    for paper_index in paper_list:
                        if finished_paper.__contains__(paper_index) is False:
                            f.write(str(paper_index) + '\t' + self.paper2title[paper_index] + '\n')
                            finished_paper[paper_index] = 1
                            cnt += 1

            f.close()
        print(cnt)

    def load_title_sim_feature(self):
        with open("../dataset/doc2vec_result.txt", encoding="utf-8") as f:
            for line in f:
                id = int(line.split('\t')[0])
                vec = line.split('\t')[1].strip().split(' ')
                vec = np.array([float(x) for x in vec], dtype=np.float64)
                self.paperid2vec[id] = vec
            f. close()

    def get_title_sim_feature(self, edge):
        node1 = self.nodes[edge.node1]
        node2 = self.nodes[edge.node2]
        paper1s = node1.get_papers_before(edge.years[-1] + 1)
        paper2s = node2.get_papers_before(edge.years[0])

        n1_vec = np.zeros(50)
        for paper in paper1s:
            n1_vec += self.paperid2vec[paper]
        n2_vec = np.zeros(50)
        for paper in paper2s:
            n2_vec += self.paperid2vec[paper]

        n1_vec /= len(paper1s)
        n2_vec /= len(paper2s)

        ans = self.cos_sim(n1_vec, n2_vec)
        return ans

    def cos_sim(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        cos = np.dot(a, b) / (a_norm * b_norm)
        return cos

    def get_authors_context_feature(self):
        paperindex2title = {}
        with open('../dataset/paper_titles_after_process_abs.txt', encoding='utf-8') as f:
            for line in f:
                content = line.strip().split('\t')
                if len(content) > 1:
                    paperindex2title[int(content[0])] = content[1]
                else:
                    paperindex2title[int(content[0])] = ""
            f.close()

        with open('../dataset/author_context_feature_abs.txt', 'w', encoding='utf-8') as f:
            model = gensim.models.doc2vec.Doc2Vec.load("model1.txt")
            n_tar = 50
            for author_index in self.labeled_authors_index.keys():
                tar_vec = np.empty(shape=[0, n_tar], dtype=np.float64)
                node = self.nodes[author_index]
                for index in range(len(node.years)):
                    if node.nyears[index] > 0:
                        input_context = ""
                        for paper_index in node.papers[index]:
                            if paperindex2title.__contains__(paper_index):
                                input_context += paperindex2title[paper_index] + ' '
                        vec = np.array(model.infer_vector(input_context.split(' '))).reshape([1, -1])
                    else:
                        vec = np.zeros((1, n_tar), dtype=np.float64)
                    tar_vec = np.append(tar_vec, vec, axis=0)

                tar_vec = tar_vec.transpose()
                res = ""
                f.write(str(author_index) + '\n')
                for i in range(tar_vec.shape[0]):
                    for j in range(tar_vec.shape[1]):
                        res += str(tar_vec[i][j]) + '\t'
                f.write(res + '\n')
            f.close()


if __name__ == "__main__":
    print("Adam Jurczy")
    print("Adam Jurczy\u0144ski")