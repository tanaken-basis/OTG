# OTG (Optimal Transport Grouping)

[English](README.en.md) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [日本語](README.jp.md)

## 概要
なるべく均一なグループ分けを自動生成するためのPythonのプログラムです。
均一性の指標の構成のために最適輸送を用いています。
最適輸送の計算には Sinkhorn-Knopp アルゴリズムを用いています。なお、3群以上へのグループ分けにおいては、multi-marginal optimal transport に対する Sinkhorn-Knopp アルゴリズム が必要になり、このプログラムにおいてもその実装を行っています。

## 使い方
- Pythonの実行環境がない場合
    - Windows の場合は、 "[otg_gradio_win.7z](https://github.com/tanaken-basis/otg/raw/main/otg_gradio_win.7z)" をダウンロードして解凍し、"otg_gradio.exe"をダブルクリックすると、グループ分けを自動生成するWebアプリが起動します。起動するまで2～3分かかる場合があります。このWebアプリケーションは、ローカルで動作します。
        - 「WindowsによってPCが保護されました」のような警告が出た場合は、「詳細情報」をクリックして、「実行」をクリックします。
    - Mac (Apple silicon) の場合は、 "[otg_gradio_mac.7z](https://github.com/tanaken-basis/otg/raw/main/otg_gradio_mac.7z)" をダウンロードして解凍し、"otg_gradio"をダブルクリックすると、グループ分けを自動生成するWebアプリが起動します。起動するまで2～3分かかる場合があります。このWebアプリケーションは、ローカルで動作します。
        - Macでアプリが実行できない場合は、ターミナルで`sudo spctl --master-disable`と入力してすべてのアプリを許可すると実行できるようになります。また、`sudo spctl --master-enable`と入力することで元の状態に戻せます。
- Pythonプログラムの使用
    - "otg_gradio.py"をPythonで実行すると、Gradioを用いて作成したWebアプリが起動します。グループ分けの自動生成のためのパラメータの調整、自動生成の計算の実行、計算結果のファイルへの出力などができます。このWebアプリケーションは、ローカルで動作させることができます。
    - "otg_flet.py"をPythonで実行すると、Fletを用いて作成したGUIアプリが起動します。グループ分けの自動生成のためのパラメータの調整、自動生成の計算の実行、計算結果のファイルへの出力などができます。
    - ノートブックのファイル"otg.ipynb"にも、同様のプログラムを記述しています。

## Pythonプログラムを使用する場合の注意点
- このコードでは、データの可視化のために、Pythonライブラリ UMAP を使用しています。UMAP のインストールについては、https://github.com/lmcinnes/umap を参照してください。
- Webアプリの作成で使用している Gradio のインストールについては、https://github.com/gradio-app/gradio を参照してください。
- アプリの作成で使用している Flet のインストールについては、https://github.com/flet-dev/flet を参照してください。

## サンプルの実行結果

#### Gradio
![alt text](otg_gradio-1.jpg)

#### Flet
![alt text](otg_flet-1.jpg)
