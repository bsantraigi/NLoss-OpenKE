import config_fix as config
from models import *
import json
import os
data = "FB15K237"
# data = "WN18RR"
os.environ['CUDA_VISIBLE_DEVICES']='1'
con = config.Config()
con.set_use_gpu(True)
con.set_in_path(f"./benchmarks/{data}/")
con.set_work_threads(12)
con.set_train_times(40)
con.set_nbatches(1000)
con.set_alpha(0.005)
con.set_bern(0)
if data == "WN18RR":
    con.set_dimension(500)
else:
    con.set_dimension(500)
con.set_margin(1.0)

if data == "WN18RR":
    con.set_ent_neg_rate(3)
elif data == "FB15K237":
    con.set_ent_neg_rate(3)
else:
    con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(20)
con.set_valid_steps(20)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir(f"./checkpoint/{data}")
con.set_result_dir(f"./result/{data}")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(DistMult)
con.train()
