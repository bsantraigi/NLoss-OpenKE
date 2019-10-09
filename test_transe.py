import config
from models import *
import json
import os
data = "WN18RR"
os.environ['CUDA_VISIBLE_DEVICES']='0'
con = config.Config()
con.set_use_gpu(True)
#Input training files from benchmarks/FB15K/ folder.
con.set_in_path(f"./benchmarks/{data}/")
#True: Input test files from the same folder.
con.set_result_dir(f"./result/{data}")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_test_model(TransE)
con.test()
