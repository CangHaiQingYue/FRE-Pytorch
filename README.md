# FRE-Pytorch
This is an Pytorch implementation of FRE network, which focus on the Edge Detection.
We did not evaluate the ODS F-score of this version, since some hyper-parameters was not fine-tuned.
The original code was implemented by TensorFLow, and we haven't made public the code yet, since our paper is not published.
However, this implementation can run without bug.

To run this model, we should change some path in ```data_loader.py   train.py  test.py``` respectively.
After that, just run 
                          ``` 
                          python train.py
                         ```
The BSDS500 dataset and NYUD dataset are available:
```
wget http://mftp.mmcheng.net/liuyun/rcf/data/bsds_pascal_train_pair.lst
wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz
```

##Citations
If you used those dataset mentioned above， please citing follow papers:
```
@inproceedings{liu2017richer,
  title={Richer Convolutional Features for Edge Detection},
  author={Liu, Yun and Cheng, Ming-Ming and Hu, Xiaowei and Wang, Kai and Bai, Xiang},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2017}
}
```
and 
```
@inproceedings{xie2015holistically,
  title={Holistically-nested edge detection},
  author={Xie, Saining and Tu, Zhuowen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1395--1403},
  year={2015}
}
```
