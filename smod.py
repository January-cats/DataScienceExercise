# -*- coding:utf-8 -*-

import pickle 
import os
import numpy as np
import pandas as pd 

# https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset


# データセットが存在するディレクトリのパスを取得（smod.pyと同じ階層にある想定）
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + '/smod.pkl' # pickleファイルのパス
key_file = dataset_dir + '/smoking_drinking_dataset_Ver01.csv' # 元データセットのファイル

# ミニデータセットのパス
save_file_mini = dataset_dir + '/smod_mini.pkl'
key_file_mini = dataset_dir + '/smoking_drinking_dataset_Ver01_mini.csv'

def _convert_numpy(key_filepath):
    dataset = {} # データセットのデータを格納する辞書

    # csvファイル全体をDataFrameに読み出し
    dataset_df = pd.read_csv(key_filepath)
    # dataframeの状態でデータの整形(Male/Famale->1/0など)
    data_sex = dataset_df['sex'] # sex列の抽出
    data_sex = np.where(data_sex.to_numpy() == 'Male', 1, 0) # NumPy配列にしてからMale/Female => 1/0へ変換
    data_drink = dataset_df['DRK_YN'] # DRK_YN列の抽出
    data_drink = np.where(data_drink.to_numpy() == 'Y', 1, 0) # Y/N => 1/0へ変換
    # dataframeの値を更新
    dataset_df['sex'] = data_sex
    dataset_df['DRK_YN'] = data_drink
    
    # [data]: 特徴量配列を格納
    head = 0 # 特徴量列の先頭
    tail = dataset_df.shape[1] - 2 # 特徴量列の先頭
    dataset['data'] = dataset_df.iloc[:, head:tail] # 特徴量配列を抽出

    # [smoke_target]: 喫煙に関する正解ラベルを格納
    dataset['smoke_target'] = dataset_df['SMK_stat_type_cd'].to_numpy()

    # [drink_target]: 飲酒に関する正解ラベルを格納
    dataset['drink_target'] = dataset_df['DRK_YN'].to_numpy()

    # 特徴量の説明を格納
    dataset['feature_names_ja'] = [
        '性別', '年齢', '身長 (cm)', '体重 (kg)', 'waistline', '視力左', '視力右', \
        '聴力左', '聴力右', 'SBP', 'DBP', 'BLDS', 'tot コレステロール', 'HDL コレステロール', 'LDL コレステロール', \
        'トリグリセリド', 'ヘモグロビン', '検尿異常(蛋白、潜血)', '血清中クレアチニン', 'SGOT ALT', 'ガンマ GTP'
        ]
    dataset['feature_names'] = [
        'sex', 'age', 'height', 'weight', 'waistline', 'sight_left', 'sight_right', \
        'hear_left', 'hear_right', 'SBP', 'DBP', 'BLDS', 'tot_chole', 'HDL_chole', 'LDL_chole', \
        'triglyceride', 'hemoglobin', 'urine_protain', 'serum_creatinine', 'SGOT ALT', 'gamma GTP'
        ]
    # 正解ラベルの説明を格納
    dataset['smoke_target_name'] = np.array(['Never', 'Used_to', 'Still'])
    dataset['drink_target_name'] = np.array(['N', 'Y'])

    return dataset

def _do_use_mini_dataset(use_mini):
    # 使うデータセットのパスを返す
    # miniデータセットのパス or オリジナルデータセットのパス
    if use_mini:
        return key_file_mini
    else:
        return key_file

def _do_use_mini_pickle(use_mini):
    # 使うpickleファイルのパスを返す
    if use_mini:
        return save_file_mini
    else:
        return save_file

def _init_smod(use_mini):
    # pickleファイルの初期化を行う
    # save_filepath = 保存する先のpickleファイルパス
    # key_filepath = 使用するデータセットのパス
    save_filepath = _do_use_mini_pickle(use_mini)
    print(save_filepath)
    key_filepath = _do_use_mini_dataset(use_mini)
    print(key_filepath)
    # dataset（辞書）の作成
    dataset = _convert_numpy(key_filepath)

    # datasetからpickleファイルを作成
    print("Creating pickle file...")
    # pickleファイルを保存する先のファイルを開く
    with open(save_filepath, 'wb') as f:
        pickle.dump(dataset, f, -1)
    # datasetをpickle化してファイルに書き込み
    print("Done!") # 初期化完了!
    return

def load_smod(data_as_df=False, use_mini=True):
    """
    smoking drinking データセットの読み込み

    params:
    --
    as_df : boolean, Default=False
        [data]をpandas.DataFrame型として抽出する（TrueでDataFrame型、FalseでNumPy配列）
    use_mini : boolean, Default=True
        Trueでオリジナルのデータセットからランダムに抽出した150個のデータで構成された標本データセットを使う
    
    returns:
    --
    : dict
    ->keys
        data : NumPy配列(or DataFrame), shape = {n_samples, n_features}
            特徴量配列
        smoke_target : NumPy配列, shape = {n_samples, }
            喫煙に関する正解ラベル
        drink_target : NumPy配列, shape = {n_samples, }
            飲酒に関する正解ラベル
        feature_names : List, shape = {n_features, }
            特徴量の名前のベクトル
        feature_names_ja : List, shape = {n_features, }
            特徴量の名前のベクトル（日本語）
        smoke_target_name : NumPy配列, shape = {n_samples, }
            正解ラベルの値がどの喫煙クラスに該当するかを書いたベクトル
        drink_target_name : NymPy配列, shape = {n_samples, }
            正解ラベルの値がどの飲酒クラスに該当するかを書いたベクトル
    """

    dataset = {} # データセットを格納する辞書
    
    save_file_path = _do_use_mini_pickle(use_mini=use_mini)

    # pickleファイルが存在するかどうかを確認
    if not os.path.exists(save_file_path):
        # もしpickleファイルが存在しないなら、pickleファイルを作成
        print("pickle file initializing...")
        _init_smod(use_mini=use_mini)

    # pickleファイルからデータセット(辞書)を読み出し
    with open(save_file_path, 'rb') as f:
        dataset = pickle.load(f)
    
    if not data_as_df:
        dataset['data'] = dataset['data'].to_numpy()

    return dataset
