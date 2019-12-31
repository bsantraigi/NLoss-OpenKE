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

test_g = create_graph(test)
print(test_g)

valid_g = create_graph(valid)
print(valid_g)

def draw_hist_n_graph(g, hist_name="minibatch_hist.svg", graph_name="rw_mini.svg"):
    out_hist = vertex_hist(g, "out")
    out_hist[0] = out_hist[0]/np.sum(out_hist[0])
    print(f"Expected Degree:{np.sum(out_hist[0]*out_hist[1][:-1])}")
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


def random_walk():
    v = train_g.vertex(random.randint(0, train_g.num_vertices() - 1))
    minib = []
    while True:
        # print("vertex:", int(v), "in-degree:", v.in_degree(), "out-degree:",
        #       v.out_degree())

        if v.out_degree() == 0:
            print("Nowhere else to go... We found the main hub!")
            break

        if len(minib) > 128:
            mini_g = Graph(directed=False)
            mini_g.add_edge_list(minib, hashed=True)

            draw_hist_n_graph(mini_g, hist_name="rwr_hist.svg", graph_name="rwr_graph.svg")
            break

        n_list = list(v.out_neighbors())
        nxt = v
        while nxt == v:
            k = random.randint(0, len(n_list)-1)
            # print(f"k: {k}")
            nxt = n_list[k]
        minib.append(train_g.edge(v, nxt))
        v = nxt


def random_walk_induced_subg():
    v = train_g.vertex(random.randint(0, train_g.num_vertices() - 1))
    minib_e = []
    minib_v = set()
    while True:
        # print("vertex:", int(v), "in-degree:", v.in_degree(), "out-degree:",
        #       v.out_degree())

        if v.out_degree() == 0:
            print("Nowhere else to go... We found the main hub!")
            break

        if len(minib_v) == 128:
            vfilt = train_g.new_vertex_property('bool')
            for v in minib_v:
                vfilt[v] = True

            mini_g = GraphView(train_g, vfilt=vfilt)
            print(mini_g)
            # mini_g = Graph(directed=False)
            # mini_g.add_edge_list(minib_e, hashed=True)

            draw_hist_n_graph(mini_g, hist_name="rwisg_hist.svg", graph_name="rwisg_graph.svg")
            break

        nxt = v
        minib_v.add(nxt)
        n_list = list(v.out_neighbors())
        while nxt == v:
            k = random.randint(0, len(n_list) - 1)
            # print(f"k: {k}")
            nxt = n_list[k]
        minib_e.append(train_g.edge(v, nxt))
        v = nxt

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

        if len(minib_e) > 128:
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


if __name__=="__main__":
    random_walk()
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