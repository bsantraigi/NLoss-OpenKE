from FastGraphSampler import *
from timeit import default_timer as timer


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def generate_train_data(g, relations, edges, triples, sampler_class, bs, data_name):
    outFile = open(f'./benchmarks/{data_name}/train2id_{sampler_class.__name__}.txt', 'w')

    s = sampler_class(g, restart_prob=0.8, minib_size=bs)

    start = timer()

    target_size = g.num_edges()
    new_train = []
    sampled_edges_undir = []
    while len(new_train) < target_size:
        minig = s.sample(bs)
        for edge in minig.get_edges():
            u, v = edge
            new_train.append((u, v, random.choice(relations[(u, v)])))
            sampled_edges_undir.append((u, v))

    # end = timer()

    remaining_edges = set(edges) - set(sampled_edges_undir)

    # print(f"Total triples before uniq: {len(new_train)}")
    # print(f"Total triples uniq: {len(f7(new_train))}")
    # print(end - start, "sec")

    '''Stage 2
    '''
    # print("Stage 2")
    # print(f"Remaining {len(remaining_edges)} edges")
    g = Graph(directed=False)
    g.add_edge_list(remaining_edges)

    # for u, v, r in new_train:
    #     e = g.edge(u,v)
    #     if e is not None:
    #         g.remove_edge(e)

    target_size = g.num_edges()
    # print(f"Stage 2: Sample from remaining G({g.num_vertices()},{g.num_edges()})")
    s = sampler_class(g, restart_prob=0.8, minib_size=bs)
    new_train2 = []
    while len(new_train2) < target_size:
        minig = s.sample(bs)
        for edge in minig.get_edges():
            u, v = edge
            new_train2.append((u, v, random.choice(relations[(u, v)])))

    end = timer()

    new_train += new_train2

    # print(f"Visited {100 * np.sum(s.visited) / g.num_vertices():0.4}% vertices.")
    # print(f"Total triples before uniq: {len(new_train)}")

    # new_train = f7(new_train)[:target_size]
    remaining_triples = list(set(triples).difference(new_train))
    print(f"[{end - start:0.4} sec]: Total triples: {len(new_train)}, uniq: {len(f7(new_train))}")

    # Add the remaining
    new_train = new_train + remaining_triples

    outFile.write(f"{len(new_train)}\n")
    for t in new_train:
        outFile.write(f"{t[0]} {t[1]} {t[2]}\n")
    outFile.close()
    # return new_train


class Gen:
    def __init__(self, data, sampler):
        self.train_g, self.train_relations, self.train_edges, self.train_triples = graph_from_txt_format(data)
        print(self.train_g)
        self.data = data
        self.sampler = sampler

    def __call__(self, bs=800):
        generate_train_data(self.train_g,
                            self.train_relations,
                            self.train_edges,
                            self.train_triples,
                            self.sampler,
                            bs,
                            self.data)


def main():
    data = "FB15K237"

    '''E[D] plots only
    '''
    data_gen = Gen(data, RWISG)
    data_gen(800)


if __name__=="__main__":
    main()
