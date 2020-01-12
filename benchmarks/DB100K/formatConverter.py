import os

entities = {}
relations = {}

src = "entities.dict"
trg = "entity2id.txt"
items = []
with open(src) as f_src:
    for line in f_src:
        items.append(line.strip().split())

with open(trg, "w") as f_trg:
    f_trg.write(f"{len(items)}\n")
    for item in items:
        entities[item[1]] = item[0]
        f_trg.write(f"{item[1]} {item[0]}\n")


src = "relations.dict"
trg = "relation2id.txt"
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


convertTriples("train.txt", "train2id.txt")
convertTriples("test.txt", "test2id.txt")
convertTriples("valid.txt", "valid2id.txt")