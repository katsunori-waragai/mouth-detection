# mouth-detection
mouth detection using OSS

## install

python3.8 のvenv 環境を立ち上げる。
その後、
```commandline
pip install mediapipe
```

## sample code
mediapipe を用いて顔のメッシュを描画するサンプルスクリプトを改造。

```
$ python face_mesh.py -h

$ python face_mesh.py /dev/video0

```

## 制限
- 一つの顔についてだけ処理する。
  - max_num_faces=2 とすれば、2個の顔が可能になる。
- どの顔が選ばれるのかは、不明。
- 顔を誤検出する。

## Done
口の開口度を算出すること。

## TODO
顔の位置を取り出すこと
- 顔の向きを算出すること
  - yaw, pitch, roll
- 口の動作状況の理解
  − 食べる用意ができている。
  - 口に入れる。
  - 咀嚼中（もぐもぐ）
  - 嚥下（飲み込む）
  - 口を閉じている。
  - 会話中
  - これらを区別する。
- 上半身の向きを基準とした顔の向き
  - 拒絶の意思表示には、「顔を横にそむける」という動作をする。
  - 

## SEE ALSO
[Face orientation angles - pitch/yaw/roll from face geometry](https://github.com/google/mediapipe/issues/2809)
