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