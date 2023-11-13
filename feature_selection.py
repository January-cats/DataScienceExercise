# -*- coding:utf-8 -*-

import sys, pathlib
sys.path.append(pathlib.Path(__file__).resolve().parents[1].as_posix())
from dataset.smod import load_smod

import numpy as np
import pandas as pd

def main():
    # 使えそうな特徴量を探す

    # データセット読み込み
    smod = load_smod(as_np=False, standardize=True)

    # smokeの状況を調べる
    # データフレームに smoke_target を追加
    # smod['data'].smoke_target = smod['smoke_target']
    smod['data'].insert(smod['data'].shape[1], 'smoke_target', smod['smoke_target']) # df.insert(追加する位置,追加column名,追加データ)

    # 喫煙状況ごとにデータを取り出し
    smoke_never = smod['data'].loc[smod['data'].smoke_target == 1] # never
    smoke_usedto = smod['data'].loc[smod['data'].smoke_target == 2]
    smoke_still = smod['data'].loc[smod['data'].smoke_target == 3]

    # still と never の比較
    
    keys = ['SBP', 'DBP', 'BLDS', 'HDL_chole', 'LDL_chole', 'hemoglobin']
    fmt = """{} stats:
[still] 平均: {}, 分散: {}
[never] 平均: {}, 分散: {}
[特徴量差] 平均: {}, 分散: {}
"""

    for key in keys:
        
        if key == 'hemoglobin':
            # 男性
            male_smoke_never = smoke_never.loc[smoke_never.sex == 1]
            male_smoke_still = smoke_still.loc[smoke_still.sex == 1]

            # still 
            still_ave = np.average(male_smoke_still[key].to_numpy()) # 平均
            still_distrib = np.var(male_smoke_still[key].to_numpy()) # 分散
            # never
            never_ave = np.average(male_smoke_never[key].to_numpy()) # 平均
            never_distrib = np.var(male_smoke_never[key].to_numpy()) # 分散
            # 特徴量差を算出
            # (still - never) / never
            ave_diff = (still_ave - never_ave) / never_ave # 平均
            distrib_diff = (still_distrib - never_distrib) / never_distrib # 分散

            # 出力
            print(fmt.format(key, still_ave, still_distrib, never_ave, never_distrib, ave_diff, distrib_diff))


            # 女性のDataFrameを抽出
            female_smoke_never = smoke_never.loc[smoke_never.sex == 0]
            female_smoke_still = smoke_still.loc[smoke_still.sex == 0]

            # still 
            still_ave = np.average(female_smoke_still[key].to_numpy()) # 平均
            still_distrib = np.var(female_smoke_still[key].to_numpy()) # 分散
            # never
            never_ave = np.average(female_smoke_never[key].to_numpy()) # 平均
            never_distrib = np.var(female_smoke_never[key].to_numpy()) # 分散
            # 特徴量差を算出
            # (still - never) / never
            ave_diff = (still_ave - never_ave) / never_ave # 平均
            distrib_diff = (still_distrib - never_distrib) / never_distrib # 分散

            # 出力
            print(fmt.format(key + "_" + "male", still_ave, still_distrib, never_ave, never_distrib, ave_diff, distrib_diff))

        
        else:
            # still 
            still_ave = np.average(smoke_still[key].to_numpy()) # 平均
            still_distrib = np.var(smoke_still[key].to_numpy()) # 分散

            # never
            never_ave = np.average(smoke_never[key].to_numpy()) # 平均
            never_distrib = np.var(smoke_never[key].to_numpy()) # 分散

            # 特徴量差を算出
            # (still - never) / never
            ave_diff = (still_ave - never_ave) / never_ave # 平均
            distrib_diff = (still_distrib - never_distrib) / never_distrib # 分散

            # 出力
            print(fmt.format(key+ "_" + "female", still_ave, still_distrib, never_ave, never_distrib, ave_diff, distrib_diff))




    return

if __name__ == '__main__':
    main()