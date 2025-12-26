# SSL の比較実験を行う

自己教師あり学習（Semi Supervised Learning） で何が良く効くかわからんので自前で実験してみる．
実装としては，

- SimCLR
- BYOL
- SimSiam
- BarlowTwins
- SwAV
- MAE

を実装してみた．

学習データセットは

- CIFAR10
- STL10
- ImageNet

を実験する予定．現状では CIFAR10で実験済み．
