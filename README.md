# SSL の比較実験を行う

自己教師あり学習（Self-Supervised Learning） で何が良く効くかわからんので自前で実験してみる．
実装としては，

- SimCLR
  `sslcompare/transforms/simclr.py` として実装
- BYOL
  `sslcompare/transforms/byol.py` として実装
- SimSiam
  `sslcompare/transforms/simsiam.py` として実装
- BarlowTwins
  `sslcompare/transforms/barlowtwins.py` として実装
- SwAV
  `sslcompare/transforms/swav.py` として実装
- MAE
  `sslcompare/transforms/mae.py` として実装

を実装してみた．

学習データセットは

- CIFAR10
  `data/cifar10` へ配置する
- STL10
  `data/stl10/` へ配置する
- ImageNet
  - 学習データは `data/imagenet/train/*.jpg` へ配置
  - テストデータは `data/imagenet/test/*.jpg` へ配置

を実験する予定．現状では CIFAR10で実験済み．
