import config_old as config
from models import *
import json
import os
import benchmark_rwx
from FastGraphSampler import *

# data = "FB15K237"
data = "WN18RR"
# data = "DB100K"

_SAMPLER_ = RWR
BS = 1400 if data == 'WN18RR' else 800
def resample_data(_data):
    data_gen = benchmark_rwx.Gen(_data, _SAMPLER_)

    def method():
        data_gen(BS)

    # return None
    return method


callback_sampler = resample_data(data)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
con = config.Config()
con.set_use_gpu(True)
con.set_in_path(f"./benchmarks/{data}/")
if callback_sampler is None:
    con.set_train_fname("train2id.txt")
else:
    callback_sampler()
    con.set_train_fname(f"train2id_{_SAMPLER_.__name__}.txt")

con.set_work_threads(12)
con.set_train_times(250)
con.set_nbatches(4000)
con.set_alpha(0.004)
con.set_bern(0)
if data == "WN18RR":
    con.set_dimension(500)
else:
    con.set_dimension(300)
con.set_margin(1.0)
con.set_ent_neg_rate(8)
# if data == "FB15K237":
#     con.set_ent_neg_rate(3)
# elif data == "WN18RR":
#     con.set_ent_neg_rate(3)
# else:
#     con.set_ent_neg_rate(1)

con.set_rel_neg_rate(0)
# con.set_opt_method("SGD")
con.set_opt_method("adagrad")
con.set_save_steps(40)
con.set_valid_steps(20)
# con.set_valid_steps(40)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir(f"./checkpoint/{data}_{_SAMPLER_.__name__}")
con.set_result_dir(f"./result/{data}_{_SAMPLER_.__name__}")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(DistMult)
con.train(callback=callback_sampler, callback_steps=10)
