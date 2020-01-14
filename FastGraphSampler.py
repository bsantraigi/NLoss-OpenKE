from graph_tool.all import *
import random
import numpy as np
import matplotlib
from collections import defaultdict


def graph_from_txt_format(data):
    rel = open(f'./benchmarks/{data}/relation2id.txt', 'r')
    ent = open(f'./benchmarks/{data}/entity2id.txt', 'r')
    data_file = open(f'./benchmarks/{data}/train2id.txt', 'r')
    # test = open(f'./benchmarks/{data}/test2id.txt', 'r')
    # valid = open(f'./benchmarks/{data}/valid2id.txt', 'r')
    # wr1 = open('./codes/train-fb.txt', 'w')

    ent_dict = {}
    rel_dict = {}
    relations = defaultdict(list)
    ents = {}

    for i, lines in enumerate(rel):
        if i == 0:
            continue
        l = lines.split()
        rel_dict[l[0]] = (l[1])

    for i, lines in enumerate(ent):
        if i == 0:
            continue
        l = lines.split()
        ent_dict[l[0]] = (l[1])
        ents[l[1]] = l[0]

    g = Graph(directed=False)
    # g.set_fast_edge_removal(fast=True)

    # Add all vertices
    g.add_vertex(n=len(ent_dict))

    edge_list = []
    triple_list = []
    for i, lines in enumerate(data_file):   #(u,v) -> {rel1, rel2, ..}
        if i == 0:
            continue
        # h, t, r
        lis = lines.split()
        u, v = int(lis[0]), int(lis[1])
        r = lis[2]
        #if u == v:
            # SKIP SELF LOOP
        #    continue
        relations[(u, v)].append(r)
        edge_list.append((u, v))
        triple_list.append((u, v, r))

    # print("# Edges:", len(edge_list))
    g.add_edge_list(edge_list)
    return g, dict(relations), edge_list, triple_list


def graph_from_dict_format(data):
    rel = open(f'./benchmarks/{data}/relations.dict', 'r')
    ent = open(f'./benchmarks/{data}/entities.dict', 'r')
    train = open(f'./benchmarks/{data}/train.txt', 'r')

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

    g = Graph(directed=False)
    # # Add all vertices
    g.add_vertex(n=len(ent_dict))

    edge_list = []
    for lines in train:   #(u,v) -> {rel1, rel2, ..}
        lis = lines.split()
        u, v = int(ent_dict[lis[0]]), int(ent_dict[lis[2]])
        if u == v:
            # SKIP SELF LOOP
            continue
        relations[(u, v)].append(lis[1])
        edge_list.append((u, v))

    print("# Edges:", len(edge_list))
    g.add_edge_list(edge_list)
    return g, dict(relations)


class FastGraphSampler:
    def __init__(self, g):
        self.batch_triples = []
        self.minib_e = set()
        self.minib_v = set()

        # Stalling is edge based
        self.stall_limit = 15
        self.stall_limit_reset = 15

        self.g = g
        self.visited = np.zeros((self.g.num_vertices()))
        self.v = self.sample_initial_vertex()

    @staticmethod
    def normalized_hist(g):
        out_hist = vertex_hist(g, "out")
        # Drop the lone vertices
        out_hist[0] = out_hist[0][1:] / np.sum(out_hist[0][1:])
        out_hist[1] = out_hist[1][1:]
        return out_hist

    def sample_initial_vertex(self):
        if np.sum(self.visited) == self.g.num_vertices():
            print(f"+", end="", flush=True)
            self.visited = np.zeros((self.g.num_vertices()))
        vl = np.where(self.visited == 0)[0]
        v = self.g.vertex(np.random.choice(vl))
        while v.out_degree() == 0:
            v = self.g.vertex(np.random.choice(vl))
        self.visited[int(v)] = 1
        self.minib_v.add(v)

        # v = self.g.vertex(random.randint(0, self.g.num_vertices() - 1))
        # while v.out_degree() == 0:
        #     v = self.g.vertex(random.randint(0, self.g.num_vertices() - 1))
        return v

    def _refresh(self):
        self.minib_e = set()
        self.minib_v = set()
        self.v = self.sample_initial_vertex()

    def sample(self, minib_size):
        # This will keep track of the subgraph size.
        # If size increase rate slows down, v will be reset.
        # The `stall_limit` kind of defines the tolerance
        # of the sampler. A high stall_limit will lead to
        # all the last remaining edges in a subgraph found
        # and being added. Lower stall_limit will jump between
        # subgraphs quickly.

        last_size = 0
        while True:
            if self.v.out_degree() == 0 or self.stall_limit == 0:
                # Raise exception or reset vertex!
                # raise Exception(f"STALLED at {self.v}. {len(self.minib_v),len(self.minib_e)}")
                if self.stall_limit == 0:
                    # print(f"STALLED at {self.v}. {len(self.minib_v), len(self.minib_e)}")
                    print(f".", end="", flush=True)
                self.v = self.sample_initial_vertex()
                self.stall_limit = self.stall_limit_reset

            nxt = self._sample_single_nxt()
            self.visited[int(nxt)] = 1
            self.stall_limit = (self.stall_limit-1) if len(self.minib_e) == last_size else self.stall_limit_reset
            last_size = len(self.minib_e)
            self.v = nxt

            if self._sample_size() >= minib_size:
                mini_g = self._pack()
                # draw_hist_n_graph(mini_g, hist_name="rwr_hist.svg", graph_name="rwr_graph.svg")
                self._refresh()
                return mini_g

    def _pack(self):
        raise NotImplementedError()

    def _sample_single_nxt(self):
        raise NotImplementedError()

    def _sample_size(self):
        raise NotImplementedError()


class RW(FastGraphSampler):
    def __init__(self, g, **kwargs):
        super(RW, self).__init__(g)

    def _pack(self):
        self.batch_triples.append(len(self.minib_e))

        efilt = self.g.new_edge_property('bool')
        for edge in self.minib_e:
            efilt[edge] = True

        # Induced Subgraph
        mini_g = GraphView(self.g, efilt=efilt)
        # print(f"Induced MiniG: ({mini_g.num_vertices()}, {mini_g.num_edges()})")

        # OLD METHOD
        # mini_g = Graph(directed=False)
        # mini_g.add_edge_list(self.minib_e, hashed=True)
        return mini_g

    def _sample_single_nxt(self):
        nxt = self.v
        n_list = list(self.v.all_neighbors())
        # while nxt == self.v:
        k = random.randint(0, len(n_list) - 1)
        nxt = n_list[k]

        self.minib_e.add(self.g.edge(self.v, nxt))
        return nxt

    def _sample_size(self):
        return len(self.minib_e)


class RWR(FastGraphSampler):
    def __init__(self, g, **kwargs):
        assert 'restart_prob' in kwargs
        super(RWR, self).__init__(g)
        self.restart_prob = kwargs['restart_prob']

        self.stall_limit = 40
        self.stall_limit_reset = 40

    def _pack(self):
        self.batch_triples.append(len(self.minib_e))

        efilt = self.g.new_edge_property('bool')
        for edge in self.minib_e:
            efilt[edge] = True

        # Induced Subgraph
        mini_g = GraphView(self.g, efilt=efilt)

        return mini_g

    def _sample_single_nxt(self):
        if random.random() < self.restart_prob and len(self.minib_v) > 0:
            # Change parent to a previously
            # discovered vertex.
            self.v = random.choice(list(self.minib_v))

        nxt = self.v
        self.minib_v.add(nxt)
        n_list = list(self.v.all_neighbors())
        # while nxt == self.v:
            # Infinite loop when v's only neighbor is v!
        k = random.randint(0, len(n_list) - 1)
        nxt = n_list[k] # This was inside the while loop

        self.minib_e.add(self.g.edge(self.v, nxt))

        return nxt

    def _sample_size(self):
        return len(self.minib_e)


class RWISG(FastGraphSampler):
    def __init__(self, g, **kwargs):
        super(RWISG, self).__init__(g)

    def _pack(self):
        vfilt = self.g.new_vertex_property('bool')
        for v in self.minib_v:
            vfilt[v] = True

        # Induced Subgraph
        mini_g = GraphView(self.g, vfilt=vfilt)
        # print(f"Induced MiniG: ({mini_g.num_vertices()}, {mini_g.num_edges()})")
        self.batch_triples.append(mini_g.num_edges())
        return mini_g

    def _sample_single_nxt(self):
        nxt = self.v
        self.minib_v.add(nxt)
        n_list = list(self.v.all_neighbors())
        # if len(n_list) == 0:
        #     raise Exception("No edge to follow! @", self.v)
        # while nxt == self.v:  # no self loop
        k = random.randint(0, len(n_list) - 1)
        nxt = n_list[k]

        self.minib_e.add(self.g.edge(self.v, nxt))

        return nxt

    def _sample_size(self):
        return len(self.minib_v)


class RWRISG(FastGraphSampler):
    def __init__(self, g, **kwargs):
        assert 'restart_prob' in kwargs
        super(RWRISG, self).__init__(g)
        self.restart_prob = kwargs['restart_prob']

        self.stall_limit = 40
        self.stall_limit_reset = 40

    def _pack(self):
        vfilt = self.g.new_vertex_property('bool')
        for v in self.minib_v:
            vfilt[v] = True

        # Induced Subgraph
        mini_g = GraphView(self.g, vfilt=vfilt)
        # print(f"Induced MiniG: ({mini_g.num_vertices()}, {mini_g.num_edges()})")
        self.batch_triples.append(mini_g.num_edges())
        return mini_g

    def _sample_single_nxt(self):
        if random.random() < self.restart_prob and len(self.minib_v) > 0:
            # Change parent to a previously
            # discovered vertex.
            self.v = random.choice(list(self.minib_v))

        nxt = self.v
        self.minib_v.add(nxt)
        n_list = list(self.v.all_neighbors())
        # while nxt == self.v:
        k = random.randint(0, len(n_list) - 1)
        nxt = n_list[k]

        self.minib_e.add(self.g.edge(self.v, nxt))

        return nxt

    def _sample_size(self):
        return len(self.minib_v)


class SimplyRandom(FastGraphSampler):
    def __init__(self, g, **kwargs):
        assert 'minib_size' in kwargs

        super(SimplyRandom, self).__init__(g)
        self.edge_list = list(self.g.edges())

        self.minib_size = kwargs['minib_size']

    def _pack(self):
        self.batch_triples.append(len(self.minib_e))
        mini_g = Graph(directed=False)
        # mini_g.add_edge_list(self.minib_e, hashed=True)
        mini_g.add_edge_list(self.minib_e)
        return mini_g

    def _sample_single_nxt(self):
        self.minib_e = random.sample(self.edge_list, self.minib_size)
        return self.minib_e[-1].target()

    def _sample_size(self):
        return len(self.minib_e)


class RWISG_NLoss(FastGraphSampler):
    def __init__(self, g, **kwargs):
        super(RWISG_NLoss, self).__init__(g)

        self.stall_limit = 15
        self.stall_limit_reset = 15

    def _pack(self):
        vfilt = self.g.new_vertex_property('bool')
        for v in self.minib_v:
            vfilt[v] = True

        # Induced Subgraph
        mini_g = GraphView(self.g, vfilt=vfilt)
        # print(f"Induced MiniG: ({mini_g.num_vertices()}, {mini_g.num_edges()})")
        self.batch_triples.append(mini_g.num_edges())

        # Get edges
        edge_stack = [mini_g.get_edges()]

        # SLOW!!!
        for v in mini_g.vertices():
            neighbors_v = self.g.get_all_edges(v)
            np.random.shuffle(neighbors_v)
            edge_stack.append(neighbors_v[:20])

        # for v in mini_g.vertices():
        #     for y, z in self.g.get_all_edges(v)[:20]:
        #         efilt[self.g.edge(y, z)] = True

        mini_g = Graph(directed=False)
        mini_g.add_edge_list(np.concatenate(edge_stack))
        # print(f"reInduced MiniG: ({mini_g.num_vertices()}, {mini_g.num_edges()})")
        return mini_g

    def _sample_single_nxt(self):
        nxt = self.v
        self.minib_v.add(nxt)
        n_list = list(self.v.all_neighbors())
        # if len(n_list) == 0:
        #     raise Exception("No edge to follow! @", self.v)
        # while nxt == self.v:  # no self loop
        k = random.randint(0, len(n_list) - 1)
        nxt = n_list[k]

        self.minib_e.add(self.g.edge(self.v, nxt))

        return nxt

    def _sample_size(self):
        return len(self.minib_v)
