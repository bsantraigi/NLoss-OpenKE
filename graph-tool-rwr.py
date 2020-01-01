from graph_tool.all import *
# from graph_tool.draw import graph_draw
from collections import defaultdict
from pylab import *
import random

'''TODO:
1. Histograms (Side by side)
2. Expected Degree of whole KG (in ED vs BS)
3. Draw Graphs (With proper layouts)
4. Data sampling code for RotatE: with replacement=False/True
5. Increase number of batches for all estimates
'''

data = "DB100K"

rel = open(f'./benchmarks/{data}/relations.dict', 'r')
ent = open(f'./benchmarks/{data}/entities.dict', 'r')
train = open(f'./benchmarks/{data}/train.txt', 'r')
test = open(f'./benchmarks/{data}/test.txt', 'r')
valid = open(f'./benchmarks/{data}/valid.txt', 'r')
# wr1 = open('./codes/train-fb.txt', 'w')

ent_dict = {}
rel_dict = {}
relations = defaultdict(list)
ents = {}

for lines in rel:
    l = lines.split()
    rel_dict[l[1]] = (l[0])
for lines in ent:
    l = lines.split()
    ent_dict[l[1]] = (l[0])
    ents[l[0]] = l[1]


def create_graph(dataFileHandle):
    g = Graph(directed=False)
    # # Add all vertices
    g.add_vertex(n=len(ent_dict))

    edge_list = []
    for lines in dataFileHandle:   #(u,v) -> {rel1, rel2, ..}
        lis = lines.split()
        relations[(int(ent_dict[lis[0]]), int(ent_dict[lis[2]]))].append(lis[1])
        edge_list.append((int(ent_dict[lis[0]]), int(ent_dict[lis[2]])))

    print("# Edges:", len(edge_list))
    g.add_edge_list(edge_list)
    return g




def normalized_hist(g):
    out_hist = vertex_hist(g, "out")
    out_hist[0] = out_hist[0] / np.sum(out_hist[0])
    return out_hist


def draw_hist_n_graph(g, hist_name="minibatch_hist.svg", graph_name="rw_mini.svg"):
    out_hist = normalized_hist(g)

    y = out_hist[0]

    figure(figsize=(6, 4))
    bar(out_hist[1][:-1], out_hist[0])
    # errorbar(out_hist[1][:-1], out_hist[0], fmt="o", yerr=err,
    #          label="in")
    # gca().set_yscale("log")
    # gca().set_xscale("log")
    # gca().set_ylim(1e-1, 1e5)
    gca().set_xlim(0, 30)
    subplots_adjust(left=0.2, bottom=0.2)
    xlabel("$k_{in}$")
    ylabel("$NP(k_{in})$")
    tight_layout()
    savefig(f"plots/{hist_name}")
    # pos = fruchterman_reingold_layout(g)
    graph_draw(g, output_size=(1000, 1000), vertex_color=[1, 1, 1, 0],
               vertex_size=2.5, edge_pen_width=0.6,
               vcmap=matplotlib.cm.gist_heat_r, output=f"plots/{graph_name}")


def sample_initial_vertex(g):
    v = g.vertex(random.randint(0, g.num_vertices() - 1))
    while v.out_degree() == 0:
        v = g.vertex(random.randint(0, g.num_vertices() - 1))
    return v


class Sampler:
    def __init__(self, g):
        self.g = g
        self.v = sample_initial_vertex(g)

        self.batch_triples = []
        self.minib_e = []
        self.minib_v = set()

    def refresh(self):
        self.minib_e = []
        self.minib_v = set()
        self.v = sample_initial_vertex(self.g)

    def sample_single_batch(self, minib_size):
        stall_limit = 4
        while True:
            if self.v.out_degree() == 0 or stall_limit == 0:
                raise Exception(f"STALLED at {self.v}")

            nxt = self.sample_nxt()
            stall_limit = (stall_limit-1) if self.v == nxt else 4
            self.v = nxt

            if self.sample_size() >= minib_size:
                mini_g = self.pack()
                # draw_hist_n_graph(mini_g, hist_name="rwr_hist.svg", graph_name="rwr_graph.svg")
                self.refresh()
                return mini_g

    def remember_the_name(self, minib_size, nbatches):
        print("Batches to sample:", nbatches)
        max_d = 0
        smooth_hist = []
        self.batch_triples = []

        while nbatches > 0:
            mini_g = self.sample_single_batch(minib_size)
            print(f"Minibatch >>  ({mini_g.num_vertices()}, {mini_g.num_edges()})")

            # Smooth histogram
            out_hist = normalized_hist(mini_g)
            max_d = max(max_d, out_hist[1][-2] + 1)
            smooth_hist.append(out_hist)

            nbatches -= 1

        # Average the histogram
        all_hist = np.zeros((len(smooth_hist), int(max_d)))
        print("Number of batches collected:", len(smooth_hist))
        for i, d_hist in enumerate(smooth_hist):
            all_hist[i, d_hist[1][:-1]] = d_hist[0]

        smooth_hist = np.median(all_hist, 0)

        ed = np.sum(smooth_hist * np.arange(smooth_hist.shape[0]))
        print(f"Expected Degree:{ed}")

        return np.mean(self.batch_triples), ed, smooth_hist

    def pack(self):
        raise NotImplementedError()

    def sample_nxt(self):
        raise NotImplementedError()

    def sample_size(self):
        raise NotImplementedError()


class RW(Sampler):
    def __init__(self, g, **kwargs):
        super(RW, self).__init__(g)

    def pack(self):
        self.batch_triples.append(len(self.minib_e))
        mini_g = Graph(directed=False)
        mini_g.add_edge_list(self.minib_e, hashed=True)
        return mini_g

    def sample_nxt(self):
        nxt = self.v
        n_list = list(self.v.out_neighbors())
        while nxt == self.v:
            k = random.randint(0, len(n_list) - 1)
            nxt = n_list[k]
        self.minib_e.append(self.g.edge(self.v, nxt))
        return nxt

    def sample_size(self):
        return len(self.minib_e)


class RWR(Sampler):
    def __init__(self, g, **kwargs):
        assert 'restart_prob' in kwargs
        super(RWR, self).__init__(g)
        self.restart_prob = kwargs['restart_prob']

    def pack(self):
        self.batch_triples.append(len(self.minib_e))
        mini_g = Graph(directed=False)
        mini_g.add_edge_list(self.minib_e, hashed=True)
        # print(f"RWR Minibatch: ({mini_g.num_vertices()}, {mini_g.num_edges()})")
        return mini_g

    def sample_nxt(self):
        if random.random() < self.restart_prob and len(self.minib_v) > 0:
            # Change parent to a previously
            # discovered vertex.
            self.v = random.choice(list(self.minib_v))

        nxt = self.v
        self.minib_v.add(nxt)
        n_list = list(self.v.out_neighbors())
        while nxt == self.v:
            k = random.randint(0, len(n_list) - 1)
            nxt = n_list[k]
        self.minib_e.append(self.g.edge(self.v, nxt))

        return nxt

    def sample_size(self):
        return len(self.minib_e)


class RWISG(Sampler):
    def __init__(self, g, **kwargs):
        super(RWISG, self).__init__(g)

    def pack(self):
        vfilt = self.g.new_vertex_property('bool')
        for v in self.minib_v:
            vfilt[v] = True

        # Induced Subgraph
        mini_g = GraphView(self.g, vfilt=vfilt)
        # print(f"Induced MiniG: ({mini_g.num_vertices()}, {mini_g.num_edges()})")
        self.batch_triples.append(mini_g.num_edges())
        return mini_g

    def sample_nxt(self):
        nxt = self.v
        self.minib_v.add(nxt)
        n_list = list(self.v.out_neighbors())
        while nxt == self.v:
            k = random.randint(0, len(n_list) - 1)
            nxt = n_list[k]
        self.minib_e.append(self.g.edge(self.v, nxt))

        return nxt

    def sample_size(self):
        return len(self.minib_v)


class SimplyRandom(Sampler):
    def __init__(self, g, **kwargs):
        assert 'minib_size' in kwargs

        super(SimplyRandom, self).__init__(g)
        self.edge_list = list(self.g.edges())

        self.minib_size = kwargs['minib_size']

    def pack(self):
        self.batch_triples.append(len(self.minib_e))
        mini_g = Graph(directed=False)
        mini_g.add_edge_list(self.minib_e, hashed=True)
        return mini_g

    def sample_nxt(self):
        self.minib_e = random.sample(self.edge_list, self.minib_size)
        return self.minib_e[-1].target()

    def sample_size(self):
        return len(self.minib_e)


def ed_vs_bs(SamplerClass, ax, rng_start, rng_end, rng_step):
    bss = []
    eds = []

    nbatches = 10
    for _bs in np.arange(rng_start, rng_end, rng_step):
        print("\nTesting batch size", _bs)
        sampler = SamplerClass(train_g, minib_size=_bs, restart_prob=0.4)
        bs, ed, _hist = sampler.remember_the_name(_bs, nbatches)
        print(bs, ed)

        bss.append(bs)
        eds.append(ed)

    ax.plot(bss, eds, marker='x')
    ax.set(xlabel="Batch Size (Number of triples)", ylabel="E[D] of Minibatch Graph",
           title=f"Expected Degree of Minibatch Graphs ({data})")

if __name__=="__main__":
    train_g = create_graph(train)
    print(train_g)

    # test_g = create_graph(test)
    # print(test_g)
    #
    # valid_g = create_graph(valid)
    # print(valid_g)

    hist_full = normalized_hist(train_g)
    ed_full = np.sum(hist_full[0]*hist_full[1][:-1])

    fig, ax = plt.subplots()

    # ed_vs_bs(random_walk, ax, 30, 10000, 1000)
    print("================ SIMPLY RANDOM ==================")
    ed_vs_bs(SimplyRandom, ax, 30, 8000, 1000)
    print("================ RW ==================")
    ed_vs_bs(RW, ax, 30, 8000, 1000)
    print("================ RWR ==================")
    ed_vs_bs(RWR, ax, 30, 8000, 1000)
    print("================ RWISG ==================")
    ed_vs_bs(RWISG, ax, 10, 2500, 250)

    ax.axhline(ed_full, linestyle='--')
    ax.legend(['SR', 'RW', 'RWR', 'RWISG', 'Full KG'])

    plt.show()


# SAMPLE
'''
train_g = Graph(directed=False)
train_g.add_vertex(6)
train_g.add_edge_list([
	(0,1),
	(1,2),
	(2,3),
	(3,4),
	(2,4),
	(0,5),
	(1,5)
])

print(train_g)
graph_draw(train_g)

train_g.remove_edge(train_g.edge(1,2))
train_g.remove_edge(train_g.edge(2,3))
train_g.remove_edge(train_g.edge(2,4))
if train_g.vertex(2).out_degree() == 0:
	train_g.remove_vertex(2)

print(train_g)
'''