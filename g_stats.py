# -*- coding: utf-8 -*-
"""G-stats.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aJpzn_riitpty0gTXDEJoLOFXUAmqakA
"""

# from google.colab import drive
# drive.mount('/content/drive')

# cd drive/My\ Drive/Colab\ Notebooks/NLoss-OpenKE/

# Commented out IPython magic to ensure Python compatibility.
# from torch.utils.data import Dataset
from operator import itemgetter
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm_notebook, tnrange, tqdm
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

# %matplotlib inline

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
    print("Reading ", path)
    with open(path) as ef:
        nE = int(ef.readline())
        triples = []
        for line in ef:
            line = line.strip()
            h,t,r = line.split(" ")
            h,t,r = int(h), int(t), int(r)
            triples.append((h,r,t))

    return triples

def gettripleAdj(path):
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

def getAdjMat(path, nE):
    with open(path) as ef:
        nT = int(ef.readline())
        print(f"Num Triples: {nT}")
        forwardMat = np.zeros((nE, nE), dtype=np.int32)
        revMat = np.zeros((nE, nE), dtype=np.int32)
        for line in ef:
            line = line.strip()
            h,t,r = line.split(" ")
            h,t,r = int(h), int(t), int(r)
            forwardMat[h][t] += 1
            revMat[t][h] += 1
    return forwardMat, revMat


class AdjList():
    def __init__(self, inpath):
        # super(AdjList, self).__init__()
        self.inpath = inpath
        self.nE, self.entities = getem(f"{self.inpath}/entity2id.txt")
        self.nR, self.rels = getem(f"{self.inpath}/relation2id.txt")
        self.nT, self.adjlist, self.degrees = gettripleAdj(f"{self.inpath}/train2id.txt")
        self.triples = gettriples(f"{self.inpath}/train2id.txt")
        self.adjmat = getAdjMat(f"{self.inpath}/train2id.txt", self.nE)

    def __len__(self):
        return len(self.adjlist)

    def __getitem__(self, index: list):
        if len(index) > 1:
            getter = itemgetter(*index)
            NS = np.concatenate(getter(self.adjlist), 0)
            inv_deg = np.concatenate(getter(self.degrees))
            shuf = np.random.permutation(inv_deg.shape[0])[:len(index)*20]
            # shuf = np.random.choice(NS.shape[0], len(index)*10, replace=False)
            return NS[shuf], inv_deg[shuf]
        else:
            return self.adjlist[index[0]], self.degrees[index[0]]



# MAIN
def main(data):
    adjlist = AdjList(f"./benchmarks/{data}/")
    adjMat1, adjMat2 = adjlist.adjmat
    # print(len(adjlist))
    # print(adjlist[[2]])
    # print(adjlist[[3]])
    # print(adjlist[[2, 3]])

    out_degrees = adjMat1.sum(1)

    in_degrees = adjMat2.sum(1)

    total_degrees = in_degrees + out_degrees

    print(f"Max in/out/total degree: {in_degrees.max(), out_degrees.max(), total_degrees.max()}")
    print(f"Min in/out/total degree: {in_degrees.min(), out_degrees.min(), total_degrees.min()}")
    print(f"Avg in/out/total degree: {in_degrees.mean(), out_degrees.mean(), total_degrees.mean()}")
    print(f"Median in/out/total degree: {np.median(in_degrees), np.median(out_degrees), np.median(total_degrees)}")

    Q = lambda x: np.quantile(x, [0.25, 0.75, 0.95]).tolist()

    print(f"Quantiles (0.25, 0.75, 0.95) - In degree: {Q(in_degrees)}")
    print(f"Quantiles (0.25, 0.75, 0.95) - Out degree: {Q(out_degrees)}")
    print(f"Quantiles (0.25, 0.75, 0.95) - Total degree: {Q(total_degrees)}")

    if data == "FB15K237":
        # 272115
        xlim = 80
    elif data == "WN18RR":
        # 86835
        xlim = 20

    # PLOT 1
    plt.hist(total_degrees, bins=max(total_degrees), density=True)
    plt.xlabel("Degree of vertex")
    plt.ylabel("Prob. density")
    plt.xlim(0, xlim)
    plt.savefig(f"plots/fulldata_{data}.png", dpi=300, bbox_inches='tight')
    plt.clf()

    xlim = 20
    # minibatch degree
    def minib(bs):
        degree_hist_super = Counter()
        num_batches = 1 + len(adjlist.triples) // bs
        for s in tqdm(range(0, len(adjlist.triples), bs)):
            sub = adjlist.triples[s:(s+bs)]
            hs, ts, _ = list(zip(*sub))
            degree_batch = Counter(hs + ts).values()
            degree_hist = Counter(degree_batch)
            degree_hist_super += degree_hist

        keys = np.array(list(degree_hist_super.keys()))
        counts = np.array(list(degree_hist_super.values()))/num_batches
        counts = counts/counts.sum()

        # PLOT 2
        plt.bar(keys, counts)
        plt.xlim(0, xlim)
        plt.xlabel("Degree of vertex"); plt.ylabel("Prob. density")
        plt.savefig(f"plots/minibatch_BS({bs})_{data}.png", dpi=300, bbox_inches='tight')
        plt.clf()

    # minibatch degree with NLoss
    def minib_nloss(bs):
        degree_hist_super = Counter()
        num_batches = 1 + len(adjlist.triples) // bs
        for s in tqdm(range(0, len(adjlist.triples), bs)):
            sub = adjlist.triples[s:(s+bs)]
            hs, ts, _ = list(zip(*sub))
            
            sub2 = adjlist[hs+ts][0].tolist()
            
            sub = sub+sub2
            hs, ts, _ = list(zip(*sub))
            
            degree_batch = Counter(hs + ts).values()
            degree_hist = Counter(degree_batch)
            degree_hist_super += degree_hist
            
        keys = np.array(list(degree_hist_super.keys()))
        counts = np.array(list(degree_hist_super.values()))/num_batches
        counts = counts/counts.sum()

        # PLOT 3
        plt.bar(keys, counts)
        plt.xlim(0, xlim)
        plt.xlabel("Degree of vertex"); plt.ylabel("Prob. density")
        plt.savefig(f"plots/NLoss_minibatch_BS({bs})_{data}.png", dpi=300, bbox_inches='tight')
        plt.clf()


    if data == "FB15K237":
        # 272115
        minib(200)
        minib_nloss(200)
        minib(1000)
        minib(1000*20) # for fair comparison
        minib_nloss(1000)
        minib(2000)
        minib_nloss(2000)
    elif data == "WN18RR":
        # 86835
        minib(100)
        minib_nloss(100)
        minib(500)
        minib_nloss(500)
        minib(1000)
        minib_nloss(1000)

    # minib(1000)
    # minib_nloss(1000)
    # minib(20000)
    # minib_nloss(20000)

if __name__ == "__main__":
    main("FB15K237")
    main("WN18RR")