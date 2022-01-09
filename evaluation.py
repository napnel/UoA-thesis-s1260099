import os
import glob
import pathlib
import argparse
import pandas as pd


if __name__ == '__main__':
    local_dir = "/home/yoshiakira/DRL-Trading/experiments/expt_algo"
    algo_expt_paths = glob.glob(os.path.join(local_dir, "*__*"))
    print(algo_expt_paths)

    for expt_path in algo_expt_paths:
        expt_path = pathlib.Path(expt_path)
        performance = pd.read_csv(expt_path.joinpath())

        # 各foldの平均を取る

    # それぞれの学習曲線を１つの表にプロットする。訓練と評価
