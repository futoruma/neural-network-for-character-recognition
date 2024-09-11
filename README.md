## 手書き数字（0~9）を認識するニューラルネットワーク

### コマンド

#### モデルを訓練して保存する

```
./main -train
```

#### 指定したモデルをテストする

```
./main -test {model_name}
```

#### 指定した画像に書かれた数字を指定したモデルで予測する

```
./main -guess {p5_pgm_image_path} -model {model_name}
```

#### 指定したモデルの重みおよびバイアスの行列を出力する

```
./main -print {model_name}
```

#### 指定したモデルをレンダリングして画像ファイルとして保存する

```
./main -render {model_name}
```

### フォルダ

- `render`
  - モデルのレンダリングによって生成された画像ファイル
- `samples`
  - サンプル、予測用の画像ファイルなど
- `saved_models`
  - 保存されたモデル
- `test_data`
  - [MNIST テストデータ](https://yann.lecun.com/exdb/mnist/)
- `training_data`
  - [MNIST 訓練データ](https://yann.lecun.com/exdb/mnist/)

### 依存ライブラリ

- [olivec](https://github.com/tsoding/olive.c)
- [stb_image_write](https://github.com/nothings/stb/blob/master/stb_image_write.h)

### 参考資料

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
- [Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Machine Learning in C](https://www.youtube.com/watch?v=PGSba51aRYU)
