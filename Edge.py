#-*-coding:utf-8-*-
import bisect
import numpy as np


class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.min_year = 1970
        self.max_year = 2021
        self.years = [x for x in range(self.min_year, self.max_year + 1)]
        self.nyears = [0 for x in range(self.min_year, self.max_year + 1)]
        self.fir_sec = [0 for x in range(self.min_year, self.max_year + 1)]
        self.fir_last = [0 for x in range(self.min_year, self.max_year + 1)]

        self.st = 9999
        self.kulcs = []
        self.irs = []
        self.f3s = []
        self.sent = 0
        self.recv = 0
        self.r = 0
        self.sum = 0
        self.label = 0
        self.sim = 0

    # type == "fir_sec" or "fir_last"
    def add_paper(self, year, type):
        self.sum += 1

        if year < self.min_year or year > self.max_year:
            return
        index = year - self.min_year

        if index != -1:
            self.nyears[index] += 1

            if type == "fir_sec":
                self.fir_sec[index] += 1

            elif type == "fir_last":
                self.fir_last[index] += 1

            self.st = min(year, self.st)

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

    def set_ed(self, y):
        self.ed = y

    def set_l(self, l):
        self.l = l

    def get_dur(self):
        return self.years[-1] - self.years[0]


if __name__ == "__main__":
    e = Edge(1, 2)
    e.add_paper(2020, "fir_last")
    e.add_paper(2020, "fir_last")
    e.add_paper(2020, "fir_last")
    e.add_paper(1997, "fir_sec")
    e.add_paper(2997, "fir_sec")
    e.add_paper(1997, "fir_sec")
    e.add_paper(1997, "fir_sec")
    e.add_paper(2020, "fir_sec")
    e.add_paper(1997, "")
    e.add_paper(12020, "")
    e.add_paper(1987, "")
    e.add_paper(1987, "")
    e.add_paper(1987, "")
    e.add_paper(1949, "fir_sec")
    print(e.years)
    print(e.nyears)
    print(e.fir_sec)
    print(e.fir_last)
