# 概要
DLTSkipプログラムに関するメモ
- 辞書学習 (Dictionary Learning) を用いて時系列 (Time series) のスキップ表現を抽出するpythonプログラム
- 得られたスキップ表現を用いて時系列分類 (Time Series Classification) タスクを実施

# 動作確認環境
- Linux (Ubuntu16.04LTS)
- anaconda4.5.1
- python3.6
- spams2.6

# Linuxでの実行手順
1. [UCR Time Series Classification Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/)から時系列データを[ダウンロード](http://www.cs.ucr.edu/~eamonn/time_series_data/UCR_TS_Archive_2015.zip)
    - ダウンロードしたzipはパスワード attempttoclassify で解凍できる
    - **dataset**ディレクトリに解凍された**UCR_TS_Archive_2015**ディレクトリを置く
1. Anacondaで仮想環境を作る
    ```
    conda create -n spams python=3.6 anaconda
    source activate spams
    ```
1. spamsをインストール
    ```
    conda install -c conda-forge python-spams
    ```
1. サンプルプログラムを実行
    ```
    python sample_tsc_ucr.py
    ```
    - UCRの時系列データセットのうち，Gun_PointというMoCapデータを用いた時系列分類を行う
    - validationデータに対する分類結果が**validation_result_Gun_Point.csv**に出力される
    - testデータに対する分類結果が**test_result_Gun_Point.csv**に出力される
1. 実行結果を確認
    - 出力された結果を**validation_result_Gun_Point_sample.csv**および**test_result_Gun_Point_sample.csv**と比較
    - 同様の出力がなされていればOK

# ファイル説明
**python**ディレクトリのファイルを簡単に説明
- **data.py** 時系列データセットを読み込むためのpythonモジュール
- **comp.py** 時系列表現を抽出するためのpythonモジュール
- **tsc.py** 時系列表現を用いて時系列分類(Time Series Classification)するためのpythonモジュール
- **util.py** 汎用的な関数を提供するためのpythonモジュール
- **sample_tsc_ucr.py** UCRの時系列データセットを用いて時系列表現を抽出して時系列分類するためのpythonプログラム
