# MNIST: Batch Normalization  (DNN with 3 Hidden layer)

Test accuracy with and without batch normalization (Each points represents 100 epochs)
![N|Solid](https://github.com/Brandon-HY-Lin/deep_learning_comparision/blob/master/datasets/MNINST/dnn_3_hidden_layers_with_batch_normalization/batch_normalization_fig_1a.png?raw=true)

This work is a reproduced work of Ioffe and Szegedy's paper [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). This code only implements figure 1(a).

Some part of the code is modified from
* [R2RT's batch normalization tutorial](https://www.tensorflow.org/versions/r1.1/get_started/mnist/beginner://r2rt.com/implementing-batch-normalization-in-tensorflow.html)

# Accuracy: 0.98
# Statistics
* 3 Hidden layer with sigmoid activation function each
* Each hidden layer has 100 activation neurons
* Using GradientDescentOptimizer with learning rate=0.5
* 50k epochs, with 60 examples per mini-batch
* Run time: 9min 39sec for batch normalization case

# Log
```sh
100%|█████████████████████████████████████| 50000/50000 [09:39<00:00, 86.22it/s]
training accuracy of with_bn: 0.981
testing accuracy of with_bn: 0.98
100%|████████████████████████████████████| 50000/50000 [05:42<00:00, 146.04it/s]
training accuracy of without_bn: 0.9769
testing accuracy of without_bn: 0.9769
```


License
----

Apache License, Version 2.0 

