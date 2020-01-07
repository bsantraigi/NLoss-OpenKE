from graph_tool.all import *
import random
import numpy as np
import matplotlib


class FastGraphSampler:
    def __init__(self, g):
        self.g = g
        self.v = self.sample_initial_vertex()

        self.batch_triples = []
        self.minib_e = []
        self.minib_v = set()

        self.stall_limit = 10

    def sample_initial_vertex(self):
        v = self.g.vertex(random.randint(0, self.g.num_vertices() - 1))
        while v.out_degree() == 0:
            v = self.g.vertex(random.randint(0, self.g.num_vertices() - 1))
        return v

    @staticmethod
    def normalized_hist(g):
        out_hist = vertex_hist(g, "out")
        out_hist[0] = out_hist[0] / np.sum(out_hist[0])
        return out_hist

    def _refresh(self):
        self.minib_e = []
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
                    print(f"STALLED at {self.v}. {len(self.minib_v), len(self.minib_e)}")

                self.v = self.sample_initial_vertex()
                self.stall_limit = 10

            nxt = self._sample_single_nxt()
            self.stall_limit = (self.stall_limit-1) if self._sample_size() == last_size else 10
            last_size = self._sample_size()
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

        self.stall_limit = 10  # vertex based selection

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
        # if len(n_list) == 0:
        #     raise Exception("No edge to follow! @", self.v)
        while nxt == self.v:  # no self loop
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
        self.stall_limit = 10  # vertex based selection

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
