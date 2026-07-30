# -*- coding: utf-8 -*-
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)  # project root
import sys, os
PROGRESS_DIR = os.path.join(ROOT, "progress_TCE5")
sys.path.insert(0, PROGRESS_DIR)
os.chdir(PROGRESS_DIR)

import run_TCE5 as __orig_run
_r = __orig_run
_og = _r.gather_batch_global_feats
def _mg(g, nids, ehids, dev):
    nf, ef = _og(g, nids, ehids, dev)
    ef = ef.clone(); ef[:,:,2]=0; ef[:,:,4:132]=0
    return nf, ef
_r.gather_batch_global_feats = _mg

_orig_argv = sys.argv[1:]
sys.argv = [
    os.path.join(PROGRESS_DIR, "run_TCE5.py"),
    "--train_hg", "../data_TCE5/Hyper_train.json",
    "--val_hg",    "../data_TCE5/Hyper_val.json",
    "--test_hg",   "../data_TCE5/Hyper_test.json",
] + [
            "--mode", "head_kd",
            "--emb_dim","256","--num_layers","2","--dropout","0.2",
            "--epochs","10","--episodes_per_epoch","40",
            "--batch_size","128","--lr","0.0005","--weight_decay","0.0001",
            "--k_hop","1","--max_edges","48","--max_nodes","192",
            "--max_members_per_edge","128","--max_hes_per_node","128",
            "--proto_k","300","--proto_bs","128","--proto_m","3",
            "--proto_reduce","logsumexp","--kmeans_iters","20",
            "--proto_interval","2","--proto_ema","0.9",
            "--proto_source","trainval",
            "--tau","0.07","--head_tau","0.05","--head_wd","0.0",
            "--kd_alpha","0.3","--kd_T","3.0",
            "--class_weight_mode","inv_sqrt",
            "--logit_adj","0.08","--logit_adj_mode","sub",
            "--focal_gamma","1.0","--ladj_ramp_epochs","10","--focal_ramp_epochs","10",
            "--pair_margin","2.0","--pair_weight","0.0",
            "--supcon_w","0.0","--supcon_temp","0.2",
            "--train_eval_max","5000","--anom_tau","0.5",
            "--auto_tau","--tau_grid","1001","--min_per_class","1",
            "--hub_degree_skip","3","--use_amp",
            "--grad_clip","1.0","--warmup_ratio","0.1",
        ]
sys.argv += ["--out_dir", "./outputs_TCE5_abl_NoOpCat"] + _orig_argv

__orig_run.main()
