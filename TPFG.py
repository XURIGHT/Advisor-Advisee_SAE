#-*-coding:utf-8-*-
from Dataset import Dataset
import math


class TPFG:
    def __init__(self, dataset):
        self.dataset = dataset

    def cal_sent(self, edge):
        i = edge.node1
        j = edge.node2

        return 0

    def cal_recv(self, edge):
        i = edge.node1
        j = edge.node2

        return 0

    def main(self):
        # Initialize all sentij as log lij
        for edge in self.dataset.edges:
            edge.sent = math.log(edge.l)

        # Initialize a counter for each node counti ← |Y^-1|
        count = []
        for node in self.dataset.nodes:
            count.append(len(node.students))

        # Initialize a stack-queue Q, enqueue all the nodes x s.t. countx = ∅;
        q = []
        head = 0
        tail = -1
        for node in self.dataset.nodes:
            if count[node.index] == 0:
                q.append(node)
                tail += 1

        while q[head] != 0:
            node = q[head]
            head += 1
            for teacher in node.teachers:
                s = str(node.index) + "#" + str(teacher)
                edge = self.dataset.edges[self.dataset.get_edge(s)]
                edge.sent = self.cal_sent(edge)
                count[teacher] -= 1
                if count[teacher] == 0:
                    q.append(self.dataset.nodes[teacher])
                    tail += 1

        while tail >= 0:
            node = q[tail]
            tail -= 1
            if node.index == 0:
                s = str(0) + "#" + str(0)
                edge = self.dataset.edges[self.dataset.get_edge(s)]
                edge.recv = 0
            else:
                recvs = []
                sents = []
                for teacher in node.teachers:
                    s = str(node.index) + "#" + str(teacher)
                    edge = self.dataset.edges[self.dataset.get_edge(s)]
                    recvs.append(edge.recv)
                    sents.append(edge.sent)

                for student in node.students:
                    s = str(student) + "#" + str(node.index)
                    edge = self.dataset.edges[self.dataset.get_edge(s)]
                    edge.recv = self.cal_recv(edge)


if __name__ == "__main__":
    d = Dataset("../dataset")
    tpfg = TPFG(d)
    tpfg.main()