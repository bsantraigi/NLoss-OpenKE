from collections import defaultdict
from utils import gettriples

dataset = "FB15K237"
# dataset = "WN18RR"

train_data = gettriples(f"./benchmarks/{dataset}/train2id.txt")

def simplify_graph():
    with open(f"checkpoint/simple_graph_{dataset}.txt", "w") as f:
        for h,r,t in train_data:
            # print(h,r,t)
            f.write(f"{h} {t}\n")

simplify_graph()

###################################################################
import louvain, leidenalg
import igraph as ig

G = ig.Graph.Read_Ncol(f"checkpoint/simple_graph_{dataset}.txt", directed=True)
# partitions = louvain.find_partition(G, louvain.ModularityVertexPartition)
partitions = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, resolution_parameter=20)

# print(len(G.es))
# print(len(G.vs))
# for i, part in enumerate(partitions):
#     print(i, len(part))
# print(len(partitions[0]))
#
# exit(0)

memberships_1 = {}
rev_memberships_1 = {}
for i, part in enumerate(partitions):
    memberships_1[i] = partitions[i]
    for x in memberships_1[i]:
        rev_memberships_1[x] = i


partitions = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, resolution_parameter=100)

memberships_2 = {}
rev_memberships_2 = {}
for i, part in enumerate(partitions):
    memberships_2[i] = partitions[i]
    for x in memberships_2[i]:
        rev_memberships_2[x] = i

# For wn18rr: use range [10, 45]
min_size = 4
max_size = 20
for e in list(rev_memberships_1.keys()):
    c1 = rev_memberships_1[e]
    c2 = rev_memberships_2[e]
    if len(memberships_1[c1]) > max_size or len(memberships_1[c1]) < min_size:
        memberships_1[c1].remove(e)
        del rev_memberships_1[e]

    if len(memberships_2[c2]) > max_size or len(memberships_2[c2]) < min_size:
        memberships_2[c2].remove(e)
        del rev_memberships_2[e]

from intbitset import intbitset
C1 = intbitset()
for i, part in memberships_1.items():
    C1 = C1|intbitset(part)

print("Remaining in P1:", len(C1))

C2 = intbitset()
for i, part in memberships_2.items():
    C2 = C2|intbitset(part)
print("Remaining in P2:", len(C2))

print("Overall:", len(C1|C2))

missed = 0
tuple_memberships_1 = defaultdict(list)
tuple_memberships_2 = defaultdict(list)
orphans = []
for h,r,t in train_data:
    eh, et = h, t
    if eh in rev_memberships_1:
        tuple_memberships_1[rev_memberships_1[eh]].append((h, r, t))
    elif et in rev_memberships_1:
        tuple_memberships_1[rev_memberships_1[et]].append((h, r, t))
    elif eh in rev_memberships_2:
        tuple_memberships_2[rev_memberships_2[eh]].append((h, r, t))
    elif et in rev_memberships_2:
        tuple_memberships_2[rev_memberships_2[et]].append((h, r, t))
    else:
        missed += 1
        orphans.append((h,r,t))


print(f"Missed {missed} of {len(train_data)}")

for _, part in tuple_memberships_1.items():
    print(len(part))

with open(f"./benchmarks/{dataset}/train2id_nl.txt", "w") as f:
    f.write(f"{len(train_data)}\n")
    for _, part in tuple_memberships_1.items():
        for h,r,t in part:
            f.write(f"{h} {t} {r}\n")
    for _, part in tuple_memberships_2.items():
        for h,r,t in part:
            f.write(f"{h} {t} {r}\n")
    for h,r,t in orphans:
        f.write(f"{h} {t} {r}\n")
