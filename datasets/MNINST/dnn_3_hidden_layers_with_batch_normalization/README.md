# MNIST: Batch Normalization  (DNN with 3 Hidden layer)

This work is a reproduced work of Ioffe and Szegedy's paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). Some part of the code is modified from
* [R2RT's batch normalization tutorial](https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginner://r2rt.com/implementing-batch-normalization-in-tensorflow.html)

# Results
* Test accuracy with and without batch normalization (Each points represents 100 epochs)
![N|Solid](https://github.com/Brandon-HY-Lin/deep_learning_comparision/blob/master/datasets/MNINST/dnn_3_hidden_layers_with_batch_normalization/batch_normalization_fig_1a.png?raw=true)

* Distribution at output of last hidden layer. Represented as percentiles of 15%, 50%, 85%.
  * Without batch normalization
![N|Solid](https://github.com/Brandon-HY-Lin/deep_learning_comparision/blob/master/datasets/MNINST/dnn_3_hidden_layers_with_batch_normalization/batch_normalization_fig_1b.png?raw=true)

  * With batch normalization
![N|Solid](https://github.com/Brandon-HY-Lin/deep_learning_comparision/blob/master/datasets/MNINST/dnn_3_hidden_layers_with_batch_normalization/batch_normalization_fig_1c.png?raw=true)

# Accuracy: 0.9776
# Statistics
* 3 Hidden layer with sigmoid activation function each
* Each hidden layer has 100 activation neurons
* Using GradientDescentOptimizer with learning rate=0.5
* 50k epochs, with 60 examples per mini-batch
* Run time: 9min 39sec for batch normalization case

# Log
```sh
100%|█████████████████████████████████████| 50000/50000 [22:14<00:00, 37.48it/s]
training accuracy of with_bn: 0.9799
testing accuracy of with_bn: 0.9776
100%|████████████████████████████████████| 50000/50000 [07:01<00:00, 118.65it/s]
training accuracy of without_bn: 0.975
testing accuracy of without_bn: 0.975
```

# Statistics on Tensorboard
You can inspect the histograms and percentiles using Tensorboard by typing following command.
```sh
$ tensorboard --logdir tmp/log/without_bn/ 
```
# Package Info
* tensorflow: Version: 1.3.0

License
----

Apache License, Version 2.0 

