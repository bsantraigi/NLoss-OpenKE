from graph_tool.all import *
# from graph_tool.draw import graph_draw
from collections import defaultdict
# from pylab import *
# import random
import matplotlib.pyplot as plt
import numpy as np

from FastGraphSampler import *

'''TODO:
[x] Histograms (Side by side)
[x] Expected Degree of whole KG (in ED vs BS)
[x] Draw Graphs (With proper layouts)
[x] Increase number of batches for all estimates
[ ] Data sampling code for RotatE: with replacement=False/True
[ ] RWISG Hangs
[ ] Do the batch size experiment : WHEN DOES OUR METHOD WORKS FOR SMALL SCALE?
'''


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
    for i, lines in enumerate(data_file):   #(u,v) -> {rel1, rel2, ..}
        if i == 0:
            continue
        # h, t, r
        lis = lines.split()
        u, v = int(lis[0]), int(lis[1])
        #if u == v:
            # SKIP SELF LOOP
        #    continue
        relations[(u, v)].append(lis[2])
        edge_list.append((u, v))

    print("# Edges:", len(edge_list))
    g.add_edge_list(edge_list)
    return g, dict(relations)


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


def just_draw(sampler, minib_size, graph_name):
    mini_g = sampler.sample(minib_size)
    print(f"Minibatch >>  ({mini_g.num_vertices()}, {mini_g.num_edges()})")
    print("Got the minibatch grpah! Drawing now...")

    pos = sfdp_layout(mini_g)
    graph_draw(mini_g, pos=pos, output_size=(1000, 1000),
               vertex_size=10, edge_pen_width=0.6,
               vcmap=matplotlib.cm.gist_heat_r, output=f"plots/{graph_name}")


def hist_n_stats(sampler, minib_size, nbatches):
    print("Batches to sample:", nbatches)
    max_d = 0
    smooth_hist = []
    sampler.batch_triples = []

    while nbatches > 0:
        mini_g = sampler.sample(minib_size)
        print(f"Minibatch >>  ({mini_g.num_vertices()}, {mini_g.num_edges()})")

        # Smooth histogram
        out_hist = FastGraphSampler.normalized_hist(mini_g)
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

    return np.mean(sampler.batch_triples), ed, smooth_hist


def ed_vs_bs(SamplerClass, ax_ed, rng_start, rng_end, rng_step):
    bss = []
    eds = []

    nbatches = 5
    rp = 0.8
    for _bs in np.arange(rng_start, rng_end, rng_step):
        print("\nTesting batch size", _bs)
        sampler = SamplerClass(train_g, minib_size=_bs, restart_prob=rp)
        bs, ed, _hist = hist_n_stats(sampler, _bs, nbatches)
        print(bs, ed)

        bss.append(bs)
        eds.append(ed)

    # Plot the last hist for now. May require to plot for
    # particular batch size later!

    print(f"Restart Prob was {rp}.")
    ax_ed.plot(bss, eds, marker='x')
    ax_ed.set(xlabel="Batch Size (Number of triples)", ylabel="E[D] of Minibatch Graph",
              title=f"Expected Degree of Minibatch Graphs ({data})")


def just_hist(SamplerClass, _bs):
    nbatches = 60
    rp = 0.8
    print("\nTesting batch size", _bs)
    sampler = SamplerClass(train_g, minib_size=_bs, restart_prob=rp)
    bs, ed, _hist = sampler.hist_n_stats(_bs, nbatches)
    print(bs, ed)
    # Plot the last hist for now. May require to plot for
    # particular batch size later!

    print(f"Restart Prob was {rp}.")
    return _hist


def pencil(SamplerClass, tag, minib_size):
    print(f"\n>>> {tag} <<<\n")
    s = SamplerClass(train_g, restart_prob=0.8, minib_size=minib_size)
    s.just_draw(minib_size, f'hist_{tag}_{data}.svg', f'graph_{tag}_{data}.svg')


if __name__=="__main__":
    data = "FB15K237"
    train_g, train_relations = graph_from_txt_format(data)

    # data = "FB15k-237"
    # train_g = graph_from_dict_format(data)

    print(train_g)

    '''E[D] plots only
    '''
    hist_full = FastGraphSampler.normalized_hist(train_g)
    ed_full = np.sum(hist_full[0]*hist_full[1][:-1])

    fig, ax = plt.subplots()

    bs_max = 800
    step = 100
    print("================ SIMPLY RANDOM ==================")
    ed_vs_bs(SimplyRandom, ax, 30, bs_max, step)
    print("================ RW ==================")
    ed_vs_bs(RW, ax, 30, bs_max, step)
    print("================ RWR ==================")
    ed_vs_bs(RWR, ax, 30, bs_max, step)

    bs_max = 250
    step = 25
    print("================ RWISG ==================")
    ed_vs_bs(RWISG, ax, 10, bs_max, step)
    print("================ RWRISG ==================")
    ed_vs_bs(RWRISG, ax, 10, bs_max, step)

    ax.axhline(ed_full, linestyle='--')
    ax.legend(['SR', 'RW', 'RWR', 'RWISG', 'RWRISG', 'Full KG'])

    plt.show()


    '''HISTOGRAMS ONLY
    '''
    # fig, ax_hist = plt.subplots(figsize=(6, 6))
    #
    # w = 1.8/5
    # d = 1.2/5
    # x = -2*d
    #
    # def brr(h):
    #     global x
    #     ax_hist.bar(x + 2*np.arange(h.shape[0]), h, w, edgecolor='black')
    #     x += d
    #
    # print("================ SIMPLY RANDOM ==================")
    # _hist = just_hist(SimplyRandom, 1000)
    # brr(_hist)
    #
    # print("================ RW ==================")
    # _hist = just_hist(RW, 1000)
    # brr(_hist)
    #
    # print("================ RWR ==================")
    # _hist = just_hist(RWR, 1000)
    # brr(_hist)
    #
    # print("================ RWISG ==================")
    # _hist = just_hist(RWISG, 300)
    # brr(_hist)
    #
    # print("================ RWRISG ==================")
    # _hist = just_hist(RWRISG, 300)
    # brr(_hist)
    #
    # _hist = normalized_hist(train_g)[0]
    # brr(_hist)
    #
    # # Not including zero
    # xlim = 20
    # ax_hist.set_xlim(1, 2*xlim + 1)
    # ax_hist.legend(['SR', 'RW', 'RWR', 'RWISG', 'RWRISG', 'Full KG'])
    # ax_hist.set_xticks(2*np.arange(1, xlim+1))
    # ax_hist.set_xticklabels(list(map(str, range(1, xlim+1))))
    #
    # ax_hist.set(xlabel="Total Degree", ylabel="Probability, p(D)",
    #      title=f"Total Degree Distribution of Minibatch Graphs ({data})")
    #
    # plt.tight_layout()
    # plt.show()

    '''Draw sample graphs
    '''
    # pencil(SimplyRandom, 'sr', 200)
    # pencil(RW, 'rw', 200)
    # pencil(RWR, 'rwr', 200)
    # pencil(RWISG, 'rwisg', 100)
    # pencil(RWRISG, 'rwrisg', 100)

    # Make a separate list for each airline

    # Assign colors for each airline and the names
    # colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
    # names = ['United Air Lines Inc.', 'JetBlue Airways', 'ExpressJet Airlines Inc.'',
    #                                                      'Delta Air Lines Inc.', 'American Airlines Inc.']

    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    # plt.hist([x1, x2, x3, x4, x5], bins=int(180 / 15), normed=True,
    #          color=colors, label=names)

    # Plot formatting
    # plt.legend()
    # plt.xlabel('Delay (min)')
    # plt.ylabel('Normalized Flights')
    # plt.title('Side-by-Side Histogram with Multiple Airlines')


