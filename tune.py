#

# tune with multiprocessing

import multiprocessing
from multiprocessing import Pool, Lock, Manager
import subprocess

import numpy as np
np.random.seed(12345)

# --
# global lock!
_global_lock = Lock()
manager = multiprocessing.Manager()
Global = manager.Namespace()
Global.idx = 0
_global_log = "_stdout.log"
# --

def run_cmd(cmd: str):
    try:
        tmp_out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        n = 0
        output = str(tmp_out.decode())  # byte->str
    except subprocess.CalledProcessError as grepexc:
        n = grepexc.returncode
        output = grepexc.output
    return output

def run_one(arg_str: str):
    # --
    with _global_lock:
        cur_idx = Global.idx
        Global.idx = Global.idx + 1
        print(f"Start task {cur_idx}: {arg_str}")
    # --
    output = run_cmd(f"MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 WHICH_XP=numpy python3 classifier.py {arg_str} --model model{cur_idx}.npz")
    # --
    with _global_lock:
        print(f"End task {cur_idx}: {arg_str}")
        with open(_global_log, 'a') as fd:
            fd.write(output)
    # --

def run_them(ranges: list, ncpu: int, shuffle=True):
    # first expand get all ranges
    all_args = [""]
    for one_ranges in ranges:
        new_all_args = []
        for a in all_args:
            for a2 in one_ranges:
                new_all_args.append(a+" "+a2)
        all_args = new_all_args
    # shuffle them all
    print(f"All tasks = {len(all_args)}")
    if shuffle:
        np.random.shuffle(all_args)
    # run them
    with Pool(ncpu) as p:
        p.map(run_one, all_args)
    # --

def main():
    tune_ranges = [
        [f"--lrate {z}" for z in [0.02, 0.015, 0.01]],
        [f"--mrate {z}" for z in [0.85, 0.9]],
        [f"--accu_step {z}" for z in [1, 4, 10, 16]],
    ]
    run_them(tune_ranges, 8)

if __name__ == '__main__':
    main()
