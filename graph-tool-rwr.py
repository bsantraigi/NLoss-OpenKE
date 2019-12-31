from graph_tool.all import *
# from graph_tool.draw import graph_draw
from collections import defaultdict
from pylab import *
import random

data = "FB15k-237"
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


def random_walk(minib_e_size=128, nbatches=100):
    batch_triples = []
    smooth_hist = []
    max_d = 0

    v = train_g.vertex(random.randint(0, train_g.num_vertices() - 1))
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

            # Smooth histogram
            out_hist = normalized_hist(mini_g)
            max_d = max(max_d, out_hist[1][-2]+1)
            smooth_hist.append(out_hist)

            # draw_hist_n_graph(mini_g, hist_name="rwr_hist.svg", graph_name="rwr_graph.svg")

            # INIT NEXT ROUND!!!
            nbatches -= 1
            minib = []
            v = train_g.vertex(random.randint(0, train_g.num_vertices() - 1))

        n_list = list(v.out_neighbors())
        nxt = v
        while nxt == v:
            if len(n_list) == 0:
                nxt = n_list[0]
            else:
                k = random.randint(0, len(n_list)-1)
                # print(f"k: {k}")
                nxt = n_list[k]
        minib.append(train_g.edge(v, nxt))
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


def random_walk_with_restart():
    restart_prob = 0.4
    v = train_g.vertex(random.randint(0, train_g.num_vertices() - 1))
    minib_e = []
    minib_v = []
    while True:
        # print("vertex:", int(v), "in-degree:", v.in_degree(), "out-degree:",
        #       v.out_degree())

        if v.out_degree() == 0:
            print("Nowhere else to go... We found the main hub!")
            break

        if len(minib_e) >= 128:
            mini_g = Graph(directed=False)
            mini_g.add_edge_list(minib_e, hashed=True)

            draw_hist_n_graph(mini_g, hist_name="rw_hist.svg", graph_name="rw_graph.svg")
            break

        if random.random() < restart_prob:
            # Change parent to a previously
            # discovered vertex.
            minib_v = list(set(minib_v))
            v = random.choice(minib_v)

        nxt = v
        minib_v.append(nxt)
        n_list = list(v.out_neighbors())
        while nxt == v:
            k = random.randint(0, len(n_list) - 1)
            # print(f"k: {k}")
            nxt = n_list[k]
        minib_e.append(train_g.edge(v, nxt))
        v = nxt


def random_walk_induced_subg(minib_v_size=128, nbatches=100):
    batch_triples = []
    smooth_hist = []
    max_d = 0

    v = train_g.vertex(random.randint(0, train_g.num_vertices() - 1))
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
            vfilt = train_g.new_vertex_property('bool')
            for v in minib_v:
                vfilt[v] = True

            # Induced Subgraph
            mini_g = GraphView(train_g, vfilt=vfilt)
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
            v = train_g.vertex(random.randint(0, train_g.num_vertices() - 1))

        nxt = v
        minib_v.add(nxt)
        n_list = list(v.out_neighbors())
        while nxt == v or nxt.out_degree() < 2:
            if len(n_list) == 0:
                nxt = n_list[0]
            else:
                k = random.randint(0, len(n_list) - 1)
                # print(f"k: {k}")
                nxt = n_list[k]
        minib_e.append(train_g.edge(v, nxt))
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

    for _bs in np.arange(rng_start, rng_end, rng_step):
        print("Testing batch size", _bs)
        bs, ed, _hist = sampler(_bs, 20)
        bss.append(bs)
        eds.append(ed)

    ax.plot(bss, eds)
    ax.set(xlabel="Batch Size (Number of triples)", ylabel="E[D] of Minibatch Graph",
           title="Expected Degree of Minibatch Graphs")

if __name__=="__main__":
    fig, ax = plt.subplots()

    ed_vs_bs(random_walk, ax, 30, 10240, 500)
    ed_vs_bs(random_walk_induced_subg, ax, 10, 1024, 100)

    plt.show()
    # random_walk_with_restart()
    # random_walk_induced_subg()


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