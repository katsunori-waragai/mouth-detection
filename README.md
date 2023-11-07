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


## TODO
顔の位置を取り出すこと
口の開口度を算出すること。
