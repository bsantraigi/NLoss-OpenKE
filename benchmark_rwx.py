#!/home/bishal/miniconda3/bin/python
from graph_tool.all import *
# from graph_tool.draw import graph_draw
from collections import defaultdict
# from pylab import *
# import random
import matplotlib.pyplot as plt
import numpy as np

from FastGraphSampler import *
from graph_tool_rwr import graph_from_txt_format

'''TODO:
[x] Histograms (Side by side)
[x] Expected Degree of whole KG (in ED vs BS)
[x] Draw Graphs (With proper layouts)
[x] Increase number of batches for all estimates
[ ] Data sampling code for RotatE: with replacement=False/True
[x] RWISG Hangs
[ ] Do the batch size experiment : WHEN DOES OUR METHOD WORKS FOR SMALL SCALE?
'''


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def generate_train_data(g, relations, sampler_class, data_name):
    outFile = open(f'./benchmarks/{data_name}/train2id_{sampler_class.__name__}.txt', 'w')
    target_size = g.num_edges()
    mbs = 300
    s = sampler_class(train_g, restart_prob=0.8, minib_size=mbs)
    new_train = []

    while len(new_train) < target_size:
        minig = s.sample(mbs)
        for edge in minig.get_edges():
            u, v = edge
            new_train.append((u, v, random.choice(relations[(u, v)])))

    print(f"Total triples before uniq: {len(new_train)}")
    print(f"Total triples uniq: {len(set(new_train))}")
    print(f"Total triples uniq: {len(f7(new_train))}")
    new_train = new_train[:target_size]
    outFile.write(f"{target_size}\n")
    for t in new_train:
        outFile.write(f"{t[0]} {t[1]} {t[2]}\n")
    outFile.close()
    # return new_train


if __name__=="__main__":
    data = "FB15K237"
    train_g, train_relations = graph_from_txt_format(data)
    print(train_g)

    '''Generate Training Data
    '''
    from timeit import default_timer as timer
    start = timer()
    generate_train_data(train_g, train_relations, RWISG, data)
    end = timer()
    print(end - start)


    '''E[D] plots only
    '''
    # hist_full = FastGraphSampler.normalized_hist(train_g)
    # ed_full = np.sum(hist_full[0]*hist_full[1][:-1])
    #
    # fig, ax = plt.subplots()
    #
    # bs_max = 500
    # step = 100
    # print("================ SIMPLY RANDOM ==================")
    # ed_vs_bs(SimplyRandom, ax, 30, bs_max, step)
    # print("================ RW ==================")
    # ed_vs_bs(RW, ax, 30, bs_max, step)
    # print("================ RWR ==================")
    # ed_vs_bs(RWR, ax, 30, bs_max, step)
    #
    # bs_max = 250
    # step = 25
    # print("================ RWISG ==================")
    # ed_vs_bs(RWISG, ax, 10, bs_max, step)
    # print("================ RWRISG ==================")
    # ed_vs_bs(RWRISG, ax, 10, bs_max, step)
    #
    # ax.axhline(ed_full, linestyle='--')
    # ax.legend(['SR', 'RW', 'RWR', 'RWISG', 'RWRISG', 'Full KG'])
    #
    # plt.show()


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


