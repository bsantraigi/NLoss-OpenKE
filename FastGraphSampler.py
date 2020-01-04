from graph_tool.all import *
import random
import numpy as np
import matplotlib


class FastGraphSampler:
    def __init__(self, g):
        self.g = g
        self.v = self.sample_initial_vertex(g)

        self.batch_triples = []
        self.minib_e = []
        self.minib_v = set()

    @staticmethod
    def sample_initial_vertex(g):
        v = g.vertex(random.randint(0, g.num_vertices() - 1))
        while v.out_degree() == 0:
            v = g.vertex(random.randint(0, g.num_vertices() - 1))
        return v

    @staticmethod
    def normalized_hist(g):
        out_hist = vertex_hist(g, "out")
        out_hist[0] = out_hist[0] / np.sum(out_hist[0])
        return out_hist

    def _refresh(self):
        self.minib_e = []
        self.minib_v = set()
        self.v = self.sample_initial_vertex(self.g)

    def sample(self, minib_size):
        stall_limit = 4
        while True:
            if self.v.out_degree() == 0 or stall_limit == 0:
                raise Exception(f"STALLED at {self.v}")

            nxt = self._sample_single_nxt()
            stall_limit = (stall_limit-1) if self.v == nxt else 4
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
        mini_g = Graph(directed=False)
        mini_g.add_edge_list(self.minib_e, hashed=True)
        return mini_g

    def _sample_single_nxt(self):
        nxt = self.v
        n_list = list(self.v.out_neighbors())
        while nxt == self.v:
            k = random.randint(0, len(n_list) - 1)
            nxt = n_list[k]
        self.minib_e.append(self.g.edge(self.v, nxt))
        return nxt

    def _sample_size(self):
        return len(self.minib_e)


class RWR(FastGraphSampler):
    def __init__(self, g, **kwargs):
        assert 'restart_prob' in kwargs
        super(RWR, self).__init__(g)
        self.restart_prob = kwargs['restart_prob']

    def _pack(self):
        self.batch_triples.append(len(self.minib_e))
        mini_g = Graph(directed=False)
        mini_g.add_edge_list(self.minib_e, hashed=True)
        # print(f"RWR Minibatch: ({mini_g.num_vertices()}, {mini_g.num_edges()})")
        return mini_g

    def _sample_single_nxt(self):
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
        n_list = list(self.v.out_neighbors())
        while nxt == self.v:
            k = random.randint(0, len(n_list) - 1)
            nxt = n_list[k]
        self.minib_e.append(self.g.edge(self.v, nxt))

        return nxt

    def _sample_size(self):
        return len(self.minib_v)


class RWRISG(FastGraphSampler):
    def __init__(self, g, **kwargs):
        assert 'restart_prob' in kwargs
        super(RWRISG, self).__init__(g)
        self.restart_prob = kwargs['restart_prob']
        # self.v0 = self.v

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
        n_list = list(self.v.out_neighbors())
        while nxt == self.v:
            k = random.randint(0, len(n_list) - 1)
            nxt = n_list[k]
        self.minib_e.append(self.g.edge(self.v, nxt))

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
        mini_g.add_edge_list(self.minib_e, hashed=True)
        return mini_g

    def _sample_single_nxt(self):
        self.minib_e = random.sample(self.edge_list, self.minib_size)
        return self.minib_e[-1].target()

    def _sample_size(self):
        return len(self.minib_e)
