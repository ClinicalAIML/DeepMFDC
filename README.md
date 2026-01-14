# DeepMFDC
Official implementation of the paper DeepMFDC: End-to-End Adaptive-Basis Contrastive Deep Clustering for Multivariate Functional Data with an Application to Intraoperative ICU Risk Phenotyping


##Training
For example, to run the experiments for the simulation scenario B, you can run the code `train_case2.py`
The Python file will automatically generate the simulation datasets and run the experiments on each dataset.


##Synthetic Dataset
The simulation scenarios A and C can be found in the `simulation` file, and you can run the R code to generate the synthetic dataset in our paper. The results from traditional functional data clustering methods can also be reproduced within those R files.

The R file `semiVS.r` is the variable selection for real data clustering; you can modify it based on your own data.
