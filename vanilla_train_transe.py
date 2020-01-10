import config_old as config
from models import *
import json
import os
import benchmark_rwx
# data = "WN18RR"


def resample_data(_data):
    data_gen = benchmark_rwx.Gen(_data)

    def method():
        return
        data_gen()

    return method


data = "FB15K237"

callback_sampler = resample_data(data)
callback_sampler()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
con = config.Config()
con.set_use_gpu(True)
con.set_in_path(f"./benchmarks/{data}/")
# con.set_train_fname("train2id_RWISG.txt")
con.set_train_fname("train2id.txt")
con.set_work_threads(12)
con.set_train_times(10000)
con.set_nbatches(29)
# con.set_nbatches(1)
con.set_alpha(0.01)
con.set_bern(0)
if data=="FB15K237":
    con.set_dimension(200)
elif data=="WN18RR":
    con.set_dimension(512)
con.set_margin(1.0)
con.set_ent_neg_rate(8)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")
# con.set_opt_method("adagrad")
con.set_save_steps(40)
con.set_valid_steps(10)
# con.set_valid_steps(40)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir(f"./checkpoint/{data}")
con.set_result_dir(f"./result/{data}")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
# con.set_train_model(TransE)
con.set_train_model(TransESoftLoss)
con.train(validation_callback=callback_sampler, validation_callback_steps=1)
