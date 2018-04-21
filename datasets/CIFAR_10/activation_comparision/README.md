# Relu and Tanh Comparison

This work is a reproduced work of [Fig 1 of AlexNet paper](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf) published in 2012. In that paper, the training speed of Relu is 6 times than tanh. However, Adam optimization was announced in 2015. As a result, this work shows how fast Relu can be using Adam optimizer. I list the differences between my work and Hinton's work below.

|   | Fig 1 in AlexNet paper | This work |
| ------ | ------ |------ | 
| Layers | 4 conv layers | 2 conv + 3 fully connected | 
| Optimizer | Gradient Descent | Adam | 
| Ratio of tanh to Relu  | ~6 times| ~2 times |

### Result
![N|Solid](https://github.com/Brandon-HY-Lin/deep_learning_comparision/blob/master/datasets/CIFAR_10/activation_comparision/training_error_rate_relu_tanh.png?raw=true)

* You can check existing training history by using following code: 
```sh
$ tensorboard --logdir log_aws
```

### Architecture
The setting of my architecture is listed below:

|Layer| Name | Settings| Output Size| # Parameters|
| ------ | ------ |------ | ------ | ------ | 
| 0 | input layer| | 32x32x3| 0|
|1| conv 1| F=5, S=1, P=2 | 32x32x64 | 5x5x3x64 + 64 = 4,864|
|| max pool 1| F=3, S=2, P=1 | 16x16x64 | 0|
|2| conv 2| F=5, S=1, P=2 | 16x16x64 | 5x5x64x64 + 64 = 102,400|
|| max pool 2| F=3, S=2, P=1 | 8x8x64 | 0|
|3| Fully Connected 3| Weight=[8x8x64, 384] | 384 | (8x8x64)x384 + 384 = 1,572,864|
|4| Fully Connected 4| Weight=[384, 192] | 192 | 384x192 + 192 = 73,920|
|5| Outpu | Weight=[192, 10] | 10 | 192x10 + 10= 1,930|

* Total num of parameters: 
4,864 + 102,400 + 1,572,864 + 73,920 + 1,930 = 1,764,048

* Memory for parameters:
1,764,048 x (4 bytes) = 7,056,192 (bytes) ~= 6.7 (Mb)

* Training Platform
AWS g2.2xlarge (GRID K520)
memoryClockRate(GHz): 0.797

* Run Time:

| | Time|
|------|------|
|real|	20m42.930s|
|user|	9m26.044s|
|sys|	6m9.016s|

### Picking Learning Rate 
Learning rate is 1.05e-4 for Adam optimizer. The green line (learning rate = 1.05e-4) has better accuracy on first 100 epochs. Notice that, gray line (2.39e-5) also has good result. However, other learning rates that are closed to (2.39e-5) has bad performance. As a result, I still choose green one (1.05e-4).

![N|Solid](https://github.com/Brandon-HY-Lin/deep_learning_comparision/blob/18e20b103c4a8194ff1ba7afbbeab73d0e782bc5/datasets/CIFAR_10/activation_comparision/learning_rate_comparision.png?raw=true)

* You can check existing log by runing:
```sh
$ tensorboard --logdir log_hyper_parameters
```

* If you want to choose by youself, use the following code:
```sh
$ python choose_hyper_parameters.py
```

### Other Details
* If you want to train by youself, run the following code: 
```sh
$ python main.py
```

* Pretrained Models
Please check the folder './log_aws/relu' and './log_aws/tanh'

### Furthur Reading
[CS231n lecture note on CNN](http://cs231n.github.io/convolutional-networks/)

### Notice
The download function of CIFAR-10 is copied from [exelban's work](https://github.com/exelban/tensorflow-cifar-10/blob/master/include/data.py)

### License

Apache License, Version 2.0
