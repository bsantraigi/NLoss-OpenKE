import config_old as config
from  models import *
import json
import os 
# data = "WN18RR"
data = "FB15K237"
os.environ['CUDA_VISIBLE_DEVICES']='1'
con = config.Config()
con.set_use_gpu(True)
con.set_in_path(f"./benchmarks/{data}/")
con.set_work_threads(12)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.005)
con.set_bern(0)
con.set_dimension(500)
con.set_margin(1.0)
con.set_ent_neg_rate(5)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(40)
con.set_valid_steps(40)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(ComplEx)
con.train()
