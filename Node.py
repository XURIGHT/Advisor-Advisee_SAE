#-*-coding:utf-8-*-
import bisect
import numpy as np
import Levenshtein

class Node:
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.min_year = 1970
        self.max_year = 2021
        self.years = [x for x in range(self.min_year, self.max_year + 1)]
        self.nyears = [0 for x in range(self.min_year, self.max_year + 1)]
        self.papers = [[] for x in range(self.min_year, self.max_year + 1)]

        self.fir = [[] for x in range(self.min_year, self.max_year + 1)]
        self.sec = [[] for x in range(self.min_year, self.max_year + 1)]
        self.last = [[] for x in range(self.min_year, self.max_year + 1)]
        self.fir_dic = {}
        self.sec_dic = {}
        self.last_dic = {}

        self.students = []  # Yi ^ -1
        self.teachers = []  # Yi
        self.co_authors = {}
        self.sum = 0
        self.start_year = 9999

    # rank == "fir" or "sec" or "last"
    def add_paper(self, year, paper_id, co_authors):
        self.sum += 1
        self.start_year = min(self.start_year, year)

        if year < self.min_year or year > self.max_year:
            return
        index = year - self.min_year

        if index != -1:
            self.nyears[index] += 1
            self.papers[index].append(paper_id)

            if self.index == co_authors[0]:
                self.fir[index].append(paper_id)
                self.fir_dic[paper_id] = 1

            elif len(co_authors) > 1 and self.index == co_authors[1]:
                self.sec[index].append(paper_id)
                self.sec_dic[paper_id] = 1

            elif self.index == co_authors[-1]:
                self.last[index].append(paper_id)
                self.last_dic[paper_id] = 1

            for author in co_authors:
                if author == self.index:
                    pass
                elif self.co_authors.__contains__(author):
                    self.co_authors[author].append(paper_id)
                else:
                    self.co_authors[author] = [paper_id]

    def bin_search(self, val):
        low = 0
        high = len(self.years) - 1
        while low <= high:
            mid = (low + high) // 2
            if self.years[mid] == val:
                return mid
            elif self.years[mid] > val:
                high = mid - 1
            else:
                low = mid + 1
        return -1

    def get_nyear_before(self, year):
        ans = 0
        for index in range(len(self.years)):
            if self.years[index] < year:
                ans += self.nyears[index]
            else:
                break
        return ans

    def get_papers_before(self, year):
        ans = []
        for index in range(len(self.years)):
            if self.years[index] < year:
                ans += self.papers[index]
            else:
                break
        return ans


if __name__ == "__main__":
    n1 = Node(1, "a")
    n1.papers = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [4441643], [642933], [], [4578146, 5172904], [], [], [4859604], [], [], [], [643126], [], [], [], [], [], [], [], [], [], [], [], []]
    t1 = 0
    for i in n1.papers:
        t1 += len(i)
    n2 = Node(2, "b")
    n2.papers = [[], [1260924, 2458352, 2930448], [804549, 3149019, 4628284], [1078887, 3530003], [2408827, 2466566, 3149927], [917452, 1260890, 2973579], [803326, 1017380, 1025050, 3147759], [281248, 305217, 3529498, 3740327, 3740343], [1023032, 1178053, 2410514, 3148843], [2410953, 3740379], [1018347, 1024694, 4629032], [282881, 2411995], [2363762, 3094534, 4123994, 4124191], [2406845, 3740385, 4796402], [278572, 3093781, 4123577, 4628473, 4796767], [], [917587, 3148727, 4629192], [1029710, 1176952, 1314217, 3149751], [2363821, 2411672, 3354820, 4124050, 4183167, 4797095], [4124291, 4797073], [905299, 3776522, 4123334], [643116, 643240, 4124087, 4124302], [593229, 1021519, 2482865, 4123944, 4124262], [2409083, 4110743], [444200, 1783519, 4123999], [1190818, 2547769], [1189440, 1190891, 2363133, 2708907, 4796197], [], [642793, 1190290, 1998152, 4053341], [3928013], [642933, 3725279], [277539, 3937735, 4023204, 4023205, 4948130], [], [4182930], [], [], [386501, 2280949], [3530780], [2370378], [], [2443439], [1189835], [1989292], [3220275], [2409475], [], [], [], [], [], [], []]
    t2 = 0
    for i in n2.papers:
        t2 += len(i)
    print("shared paper: ", 642933)
    print("advisee 735955 total: ", t1)
    print("advisor 391050 total: ", t2)
