# ウェブカメラを用いてリアルタイムに静止画中の顔を動かす

大学の授業の演習で作成したプログラムです。  
顔のパーツ(口、鼻、目)に対してのみ考慮されているので、髪などの頭、周囲との整合性は取れていないので絵的には違和感のあるものになります。

## 参考論文
Hadar Averbuch-Elor, Daniel Cohen-Or, Johannes Kopf, and Michael F. Cohen. Bringing portraits to life. ACM Transactions on Graphics (Proceeding of SIGGRAPH Asia 2017), 36(6):196, 2017.  
http://cs.tau.ac.il/~averbuch1/portraitslife/index.htm

## 動作方法(コマンドラインでの操作)
- 必要なライブラリ(dlib, numpy, cv2)をインストール
```
pip install dlib numpy opencv-python
```

- 学習済みモデルをダウンロード  
  http://dlib.net/files/ から shape_predictor_68_face_landmarks.dat.bz2 をダウンロードして、解答する。

- 動かしたい静止画を用意  
  動かしたい顔が写っているファイルを用意してください。

  著作権や肖像権などの関係でサンプルファイルはこのリポジトリにはあげていません。  
  動作確認では、jpeg形式での確認をしました。

- Python3で実行  
  ここまでで用意してきたファイルをこのリポジトリのディレクトリの中に入れて、以下のコマンドを実行してください。
```
python warping_realtime.py (用意した画像ファイル)
```

- 正面を向いた顔を検出  
  カメラのアクセスを求められます。
  画面に映った自分の顔が正面を向いていることを確認してください。  
  向いていなかった場合は、正面を向いて0以外のキーを押してください。押したタイミングの顔のパーツの配置が標準位置として使用されます。  
  プログラムを開始する際は0キーを押してください。

- 動く静止画  
  口や目を動かしてみてください。静止画の画像も一緒に動くのがわかると思います。  
  終了する時には、1キーを押して終了してください。

- 動画を出力  
  以下のファイルが出力されます。
  - 静止画が動いている動画 (output.mp4)
  - 顔のパーツがどの部位で検出されているかが点描画によってわかるWebカメラから得られた動画 (pointsvideo.mp4)