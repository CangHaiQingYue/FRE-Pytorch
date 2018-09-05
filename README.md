# FRE-Pytorch
This is an Pytorch implementation of FRE network, which focus on the Edge Detection.
We did not evaluate the ODS F-score of this version, since some hyper-parameters was not fine-tuned.
The original code was implemented by TensorFLow, and we haven't made public the code yet, since our paper is not published.
However, this implementation can run without bug.

To run this model, we should change some path in ```data_loader.py   train.py  test.py``` respectively.
After that, just run ``` Python
                          python train.py
                         ````
