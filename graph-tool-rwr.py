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
#FB
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

train_g = create_graph(train)
print(train_g)

# test_g = create_graph(test)
# print(test_g)
#
# valid_g = create_graph(valid)
# print(valid_g)

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

        self.batch_triples = []

        self.v = sample_initial_vertex(g)
        self.minib_e = []
        self.minib_v = []

    def refresh(self):
        self.minib_e = []
        self.minib_v = []
        self.v = sample_initial_vertex(self.g)

    def sample_single_batch(self, minib_e_size):
        stall_limit = 4
        while True:
            if self.v.out_degree() == 0 or stall_limit == 0:
                raise Exception(f"STALLED at {self.v}")

            nxt = self.sample_nxt()
            stall_limit = (stall_limit-1) if self.v == nxt else 4
            self.v = nxt

            if len(self.minib_e) >= minib_e_size:
                mini_g = self.pack()
                # draw_hist_n_graph(mini_g, hist_name="rwr_hist.svg", graph_name="rwr_graph.svg")
                self.refresh()
                return mini_g

    def remember_the_name(self, minib_size, nbatches):
        # def random_walk_with_restart(g, minib_size=128, nbatches=100, restart_prob=0.4):

        print("Batches to sample:", nbatches)
        max_d = 0
        smooth_hist = []

        while nbatches > 0:
            mini_g = self.sample_single_batch(minib_size)

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

        smooth_hist = np.mean(all_hist, 0)

        ed = np.sum(smooth_hist * np.arange(smooth_hist.shape[0]))
        print(f"Expected Degree:{ed}")

        return np.mean(self.batch_triples), ed, smooth_hist

    def pack(self):
        raise NotImplementedError()

    def sample_nxt(self):
        raise NotImplementedError()

class RWR(Sampler):
    def __init__(self, g, restart_prob = 0.4):
        super(RWR, self).__init__(g)
        self.restart_prob = restart_prob

    def pack(self):
        self.batch_triples.append(len(self.minib_e))
        mini_g = Graph(directed=False)
        mini_g.add_edge_list(self.minib_e, hashed=True)
        print(f"RWR Minibatch: ({mini_g.num_vertices()}, {mini_g.num_edges()})")
        return mini_g

    def sample_nxt(self):
        if random.random() < self.restart_prob and len(self.minib_v) > 0:
            # Change parent to a previously
            # discovered vertex.
            self.minib_v = list(set(self.minib_v))
            self.v = random.choice(self.minib_v)

        nxt = self.v
        self.minib_v.append(nxt)
        n_list = list(self.v.out_neighbors())
        while nxt == self.v:
            # if len(n_list) == 1:
            #     nxt = n_list[0]
            # else:
            k = random.randint(0, len(n_list) - 1)
            # print(f"k: {k}")
            nxt = n_list[k]
        self.minib_e.append(self.g.edge(self.v, nxt))

        return nxt


def simply_random(g, minib_e_size=128, nbatches=100):
    batch_triples = []
    smooth_hist = []
    max_d = 0

    print("Batches to sample:", nbatches)
    edge_list = list(g.edges())
    while nbatches > 0:
        # print("vertex:", int(v), "in-degree:", v.in_degree(), "out-degree:",
        #       v.out_degree(), "minib:", len(minib))

        minib = random.sample(edge_list, minib_e_size)
        batch_triples.append(len(minib))
        mini_g = Graph(directed=False)
        mini_g.add_edge_list(minib, hashed=True)
        print(f"Sampled Minibatch: ({mini_g.num_vertices()}, {mini_g.num_edges()})")

        # Smooth histogram
        out_hist = normalized_hist(mini_g)
        max_d = max(max_d, out_hist[1][-2]+1)
        smooth_hist.append(out_hist)

        # draw_hist_n_graph(mini_g, hist_name="rwr_hist.svg", graph_name="rwr_graph.svg")

        # INIT NEXT ROUND!!!
        nbatches -= 1

    # Average the histogram
    all_hist = np.zeros((len(smooth_hist), int(max_d)))
    print("Number of batches collected:", len(smooth_hist))
    for i, d_hist in enumerate(smooth_hist):
        all_hist[i, d_hist[1][:-1]] = d_hist[0]

    smooth_hist = np.mean(all_hist, 0)

    ed = np.sum(smooth_hist * np.arange(smooth_hist.shape[0]))
    print(f"Expected Degree:{ed}")

    return np.mean(batch_triples), ed, smooth_hist


def random_walk(g, minib_e_size=128, nbatches=100):
    batch_triples = []
    smooth_hist = []
    max_d = 0

    v = sample_initial_vertex(g)
    minib = []
    print("Batches to sample:", nbatches)
    while nbatches > 0:
        # print("vertex:", int(v), "in-degree:", v.in_degree(), "out-degree:",
        #       v.out_degree(), "minib:", len(minib))

        if v.out_degree() == 0:
            print("Nowhere else to go... We found the main hub!")
            break

        if len(minib) >= minib_e_size:
            batch_triples.append(len(minib))
            mini_g = Graph(directed=False)
            mini_g.add_edge_list(minib, hashed=True)
            print(f"RW Minibatch: ({mini_g.num_vertices()}, {mini_g.num_edges()})")

            # Smooth histogram
            out_hist = normalized_hist(mini_g)
            max_d = max(max_d, out_hist[1][-2]+1)
            smooth_hist.append(out_hist)

            # draw_hist_n_graph(mini_g, hist_name="rwr_hist.svg", graph_name="rwr_graph.svg")

            # INIT NEXT ROUND!!!
            nbatches -= 1
            minib = []
            v = sample_initial_vertex(g)

        nxt = v
        n_list = list(v.out_neighbors())
        while nxt == v:
            # if len(n_list) == 1:
            #     nxt = n_list[0]
            # else:
            k = random.randint(0, len(n_list)-1)
            # print(f"k: {k}")
            nxt = n_list[k]
        minib.append(g.edge(v, nxt))
        v = nxt

    # Average the histogram
    all_hist = np.zeros((len(smooth_hist), int(max_d)))
    print("Number of batches collected:", len(smooth_hist))
    for i, d_hist in enumerate(smooth_hist):
        all_hist[i, d_hist[1][:-1]] = d_hist[0]

    smooth_hist = np.mean(all_hist, 0)

    ed = np.sum(smooth_hist * np.arange(smooth_hist.shape[0]))
    print(f"Expected Degree:{ed}")

    return np.mean(batch_triples), ed, smooth_hist


def random_walk_induced_subg(g, minib_v_size=128, nbatches=100):
    batch_triples = []
    smooth_hist = []
    max_d = 0

    v = sample_initial_vertex(g)
    minib_e = []
    minib_v = set()
    print("Batches to sample:", nbatches)
    while nbatches > 0:
        # print("vertex:", int(v), "in-degree:", v.in_degree(), "out-degree:",
        #       v.out_degree())

        if v.out_degree() == 0:
            print("Nowhere else to go... We found the main hub!")
            break

        if len(minib_v) >= minib_v_size:
            vfilt = g.new_vertex_property('bool')
            for v in minib_v:
                vfilt[v] = True

            # Induced Subgraph
            mini_g = GraphView(g, vfilt=vfilt)
            print(f"Induced MiniG: ({mini_g.num_vertices()}, {mini_g.num_edges()})")
            batch_triples.append(mini_g.num_edges())

            # Smooth histogram
            out_hist = normalized_hist(mini_g)
            max_d = max(max_d, out_hist[1][-2] + 1)
            smooth_hist.append(out_hist)

            # draw_hist_n_graph(mini_g, hist_name="rwisg_hist.svg", graph_name="rwisg_graph.svg")

            # INIT NEXT ROUND!!!
            nbatches -= 1
            minib_e = []
            minib_v = set()
            v = sample_initial_vertex(g)

        nxt = v
        minib_v.add(nxt)
        n_list = list(v.out_neighbors())
        while nxt == v:
            # if len(n_list) == 1:
            #     nxt = n_list[0]
            # else:
            k = random.randint(0, len(n_list) - 1)
            # print(f"k: {k}")
            nxt = n_list[k]
        minib_e.append(g.edge(v, nxt))
        v = nxt

    # Average the histogram
    all_hist = np.zeros((len(smooth_hist), int(max_d)))
    print("Number of batches collected:", len(smooth_hist))
    for i, d_hist in enumerate(smooth_hist):
        all_hist[i, d_hist[1][:-1]] = d_hist[0]

    smooth_hist = np.mean(all_hist, 0)

    ed = np.sum(smooth_hist * np.arange(smooth_hist.shape[0]))
    print(f"Expected Degree:{ed}")

    return np.mean(batch_triples), ed, smooth_hist


def ed_vs_bs(sampler, ax, rng_start, rng_end, rng_step):
    bss = []
    eds = []

    nbatches = 10
    for _bs in np.arange(rng_start, rng_end, rng_step):
        print("Testing batch size", _bs)
        bs, ed, _hist = sampler.remember_the_name(_bs, nbatches)
        bss.append(bs)
        eds.append(ed)

    ax.plot(bss, eds)
    ax.set(xlabel="Batch Size (Number of triples)", ylabel="E[D] of Minibatch Graph",
           title=f"Expected Degree of Minibatch Graphs ({data})")

if __name__=="__main__":
    rwr = RWR(train_g)
    fig, ax = plt.subplots()

    # ed_vs_bs(simply_random, ax, 30, 10000, 1000)
    # ed_vs_bs(random_walk, ax, 30, 10000, 1000)
    ed_vs_bs(rwr, ax, 30, 2000, 500)
    # ed_vs_bs(random_walk_induced_subg, ax, 10, 2000, 200)

    ax.legend(['SR', 'RW', 'RWR', 'RWISG'])
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