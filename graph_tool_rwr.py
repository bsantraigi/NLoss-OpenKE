from graph_tool.all import *
# from graph_tool.draw import graph_draw
from collections import defaultdict
# from pylab import *
# import random
import matplotlib.pyplot as plt
import numpy as np

from FastGraphSampler import *


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

    # Degree 0 is ignored
    ed = np.sum(smooth_hist * (1+np.arange(smooth_hist.shape[0])))
    print(f"Expected Degree:{ed}")

    return np.mean(sampler.batch_triples), ed, smooth_hist


def ed_vs_bs(SamplerClass, plt_axis_ed, rng_start, rng_end, rng_step):
    edge_bss = []
    eds = []
    v_bss = []

    nbatches = 5
    rp = 0.8
    for _bs in np.arange(rng_start, rng_end, rng_step):
        v_bss.append(_bs)
        print(f"\n[{SamplerClass.__name__}]Testing batch size", _bs)
        sampler = SamplerClass(train_g, minib_size=_bs, restart_prob=rp)
        bs, ed, _hist = hist_n_stats(sampler, _bs, nbatches)
        print(bs, ed)

        edge_bss.append(bs)
        eds.append(ed)

    # Plot the last hist for now. May require to plot for
    # particular batch size later!

    print(f"Restart Prob was {rp}.")
    plt_axis_ed.plot(edge_bss, eds, marker='x')
    for bs, ed, vn in zip(edge_bss, eds, v_bss):
        plt_axis_ed.annotate(str(vn), xy=(bs, ed))
    plt_axis_ed.set(xlabel="Batch Size (Number of triples)", ylabel="E[D] of Minibatch Graph",
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
    # data = "WN18RR"
    # train_g, train_relations = graph_from_dict_format(data)

    data = "FB15K237"
    train_g, train_relations, _, _ = graph_from_txt_format(data)

    print(train_g)

    '''E[D] plots only
    '''
    hist_full = FastGraphSampler.normalized_hist(train_g)
    ed_full = np.sum(hist_full[0]*hist_full[1][:-1])

    fig, ax = plt.subplots()

    bs_max = 1650
    step = 400
    print("================ SIMPLY RANDOM ==================")
    ed_vs_bs(SimplyRandom, ax, 30, bs_max, step)
    print("================ RW ==================")
    ed_vs_bs(RW, ax, 30, bs_max, step)
    print("================ RWR ==================")
    ed_vs_bs(RWR, ax, 30, bs_max, step)

    bs_max = 620
    step = 200
    print("================ RWISG ==================")
    ed_vs_bs(RWISG, ax, 10, bs_max, step)
    print("================ RWRISG ==================")
    ed_vs_bs(RWRISG, ax, 10, bs_max, step)

    # bs_max = 1220
    # step = 200
    print("================ RWISG_NLoss ==================")
    ed_vs_bs(RWISG_NLoss, ax, 10, bs_max, step)

    ax.axhline(ed_full, linestyle='--')
    ax.legend(['SR', 'RW', 'RWR', 'RWISG', 'RWRISG', 'RWIS+', 'Full KG'])

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


