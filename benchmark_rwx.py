from FastGraphSampler import *
from timeit import default_timer as timer
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def generate_train_data(g, relations, edges, triples, sampler_class, bs, data_name):
    print(sampler_class.__name__, end=" ", flush=True)
    s = sampler_class(g, restart_prob=0.8, minib_size=bs)

    start = timer()

    target_size = g.num_edges()
    new_train = []

    L = 0
    while L < target_size:
        minig = s.sample(bs)
        L += minig.num_edges()
        new_train.append(minig.get_edges())

    sampled_edges_undir = [(u, v) if (u, v) in relations
                           else (v, u) for u, v in np.concatenate(new_train)]
    new_train = [(u, v, random.choice(relations[(u, v)])) for u, v in sampled_edges_undir]

    remaining_edges = set(edges) - set(sampled_edges_undir)

    # print(f"Total triples before uniq: {len(new_train)}")
    # print(f"Total triples uniq: {len(f7(new_train))}")
    # end = timer()
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

    target_size = g.num_edges() # 2* is causing heavy oversampling

    # print(f"Stage 2: Sample from remaining G({g.num_vertices()},{g.num_edges()})")
    s = sampler_class(g, restart_prob=0.8, minib_size=bs)
    new_train2 = []
    L = 0
    while L < target_size:
        minig = s.sample(bs)
        L += minig.num_edges()
        new_train2.append(minig.get_edges())

    new_train2 = [(u, v) if (u, v) in relations else (v, u) for (u, v) in np.concatenate(new_train2)]
    new_train2 = [(u, v, random.choice(relations[(u, v)])) for u, v in new_train2]

    new_train += new_train2

    # print(f"Visited {100 * np.sum(s.visited) / g.num_vertices():0.4}% vertices.")
    # print(f"Total triples before uniq: {len(new_train)}")

    # new_train = f7(new_train)[:target_size]
    remaining_triples = list(set(triples).difference(new_train))
    end = timer()
    print(f"[{end - start:0.4} sec]: Total triples: {len(new_train)}, uniq: {len(f7(new_train))}, remains: {len(remaining_triples)}")

    # Add the remaining
    new_train = new_train + remaining_triples

    # write_to_file(data_name, new_train, sampler_class)
    return new_train


def write_to_file(data_name, new_train, sampler_class):
    outFile = open(f'./benchmarks/{data_name}/train2id_{sampler_class.__name__}.txt', 'w')
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

    def __call__(self, bs):
        """
        Gets samples for the whole dataset and
        :param bs:
        :return:
        """
        new_train = generate_train_data(self.train_g,
                                        self.train_relations,
                                        self.train_edges,
                                        self.train_triples,
                                        self.sampler,
                                        bs,
                                        self.data)
        write_to_file(self.data, new_train, self.sampler)

    def get_sampled_data(self, bs):
        return generate_train_data(self.train_g,
                                   self.train_relations,
                                   self.train_edges,
                                   self.train_triples,
                                   self.sampler,
                                   bs,
                                   self.data)


def main(data, sampler_class, bs):


    '''E[D] plots only
    '''
    data_gen = Gen(data, sampler_class)
    new_train = data_gen.get_sampled_data(bs)

    triple_count = Counter(new_train)
    print(f"{len(triple_count)} triples in list")

    fig, ax = plt.subplots()
    ax.plot(sorted(triple_count.values(), reverse=True))
    ax.set(xlabel="Ranked triple", ylabel="Repetition Count", yscale="log",
           title=f"Sampling Algorithm: {data_gen.sampler.__name__}/{data}/BS_{bs}")
    ax.set_ylim(top=100)
    fig.savefig(f"plots/oversampling/{data}_{data_gen.sampler.__name__}_BS_{bs}.svg")


if __name__=="__main__":
    # main(data="WN18RR", sampler_class=RWISG_NLoss, bs=300)
    # main(data="FB15K237", sampler_class=RWISG_NLoss, bs=300)
    # main(data="DB100K", sampler_class=RWISG_NLoss, bs=300)
    #
    # main(data="WN18RR", sampler_class=RWISG, bs=300)
    # main(data="FB15K237", sampler_class=RWISG, bs=300)
    # main(data="DB100K", sampler_class=RWISG, bs=300)
    #
    # main(data="WN18RR", sampler_class=RWRISG, bs=300)
    # main(data="FB15K237", sampler_class=RWRISG, bs=300)
    # main(data="DB100K", sampler_class=RWRISG, bs=300)

    main(data="WN18RR", sampler_class=RWR, bs=2500)
    main(data="FB15K237", sampler_class=RWR, bs=2500)
    main(data="DB100K", sampler_class=RWR, bs=2500)

    main(data="WN18RR", sampler_class=RW, bs=300)
    main(data="FB15K237", sampler_class=RW, bs=300)
    main(data="DB100K", sampler_class=RW, bs=300)


