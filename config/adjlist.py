from intbitset import intbitset
from torch.utils.data import Dataset
from operator import itemgetter
import numpy as np

def getem(path):
    print("Reading ", path)
    with open(path) as ef:
        nE = int(ef.readline())
        print(f"Num Items: {nE}")
        entities = {}
        for line in ef:
            line = line.strip()
            name,id = line.split("\t")
            entities[id] = name

    return nE, entities

def gettriples(path):
    with open(path) as ef:
        nE = int(ef.readline())
        print(f"Num Triples: {nE}")
        adjlist = [list() for _ in range(nE)]
        for line in ef:
            line = line.strip()
            h,t,r = line.split(" ")
            h,t,r = int(h), int(t), int(r)
            adjlist[h].append([h, t, r])
            adjlist[t].append([h, t, r])

        degrees = [0]*nE
        for i in range(nE):
            adjlist[i] = np.array(adjlist[i], dtype=np.int).reshape(-1, 3)
            # degrees[i] = np.array([1] * adjlist[i].shape[0])
            if adjlist[i].shape[0] > 0:
                degrees[i] = np.array([1/adjlist[i].shape[0]]*adjlist[i].shape[0], dtype=np.float32)
            else:
                degrees[i] = np.array([1] * adjlist[i].shape[0], dtype=np.float32)

    return nE, adjlist, degrees


class AdjList(Dataset):
    def __init__(self, inpath):
        super(AdjList, self).__init__()
        self.inpath = inpath
        self.nE, self.entities = getem(f"{self.inpath}/entity2id.txt")
        self.nR, self.rels = getem(f"{self.inpath}/relation2id.txt")
        self.nT, self.adjlist, self.degrees = gettriples(f"{self.inpath}/train2id.txt")

    def __len__(self):
        return len(self.adjlist)

    def __getitem__(self, index: list):
        if len(index) > 1:
            getter = itemgetter(*index)
            return np.concatenate(getter(self.adjlist), 0), np.concatenate(getter(self.degrees))
        else:
            return self.adjlist[index[0]], self.degrees[index[0]]

if __name__=="__main__":
    adjlist = AdjList("../benchmarks/FB15K237/")
    print(len(adjlist))
    print(adjlist[[2]])
    print(adjlist[[3]])
    print(adjlist[[2, 3]])

