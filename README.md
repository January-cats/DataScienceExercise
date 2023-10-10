# DataScienceExercise

# 概要
「smoking_drinking_dataset」（略してsmod）を使うためのAPIです。
使い方は、3つのファイル（smod.py, smoking_drinking_dataset_Ver01.csv, smoking_drinking_dataset_Ver01_mini.csv）を同じディレクトリに入れて、smod.py中のload_smod()を呼び出すだけです。

smoking_drinking_dataset_Ver01.csvはオリジナルのデータセットで、smoking_drinking_dataset_Ver01_mini.csvはそのうち先頭150サンプルです。
※smoking_drinking_dataset_Ver01.csvは大きすぎてgithubに置けなかったので、オリジナルからダウンロードしてください。
→ https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset
テスト時など負荷軽減のためにminiデータセットをご使用ください。

load_smod()の詳細は以下のとおりです。（コード中にも同じことが書かれています。）
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
