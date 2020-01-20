import os

entities = {}
relations = {}

folder = "benchmarks/yago2/"
src = f"{folder}/entities.dict"
trg = f"{folder}/entity2id.txt"
items = []
with open(src) as f_src:
    for line in f_src:
        items.append(line.strip().split())

with open(trg, "w") as f_trg:
    f_trg.write(f"{len(items)}\n")
    for item in items:
        entities[item[1]] = item[0]
        f_trg.write(f"{item[1]} {item[0]}\n")


src = f"{folder}/relations.dict"
trg = f"{folder}/relation2id.txt"
items = []
with open(src) as f_src:
    for line in f_src:
        items.append(line.strip().split())

with open(trg, "w") as f_trg:
    f_trg.write(f"{len(items)}\n")
    for item in items:
        relations[item[1]] = item[0]
        f_trg.write(f"{item[1]} {item[0]}\n")


def convertTriples(src,trg):
    items = []
    with open(src) as f_src:
        for line in f_src:
            # h,r,t
            items.append(line.strip().split())

    with open(trg, "w") as f_trg:
        f_trg.write(f"{len(items)}\n")
        for item in items:
            # h,t,r
            f_trg.write(f"{entities[item[0]]} {entities[item[2]]} {relations[item[1]]}\n")


convertTriples(f"{folder}/train.txt", f"{folder}/train2id.txt")
convertTriples(f"{folder}/test.txt", f"{folder}/test2id.txt")
convertTriples(f"{folder}/valid.txt", f"{folder}/valid2id.txt")
