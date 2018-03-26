# Max Pooling Parameters

Comparison between 3 different parameters as shown below:

 * Non-Overlapping
    F2_S2_SAME: Filter Size = 2x2, stride=2x2, padding=SAME

 * Overlapping 
    F3_S2_SAME: Filter Size = 3x3, stride=2x2, padding=SAME

 * Overlapping with big filter (LeNet5 setting)
    LeNet5_F5_S1_VALID: Filter Size = 5x5, stride=1x1, padding=VALID

![N|Solid](https://github.com/Brandon-HY-Lin/deep_learning_comparision/blob/master/cnn/parameters/max_pool/conv_twice.png?raw=true)

The first column is original MNIST image (digit 7). The 2nd column is image after max_pool filter. The 3rd column is image after 2 sucessive max_pool filters.

The non-overlapping setting preserved more details. Moreever, when the filters increase, the image becomes more blur.

### Further Reading
[CS231n lecture note](http://cs231n.github.io/convolutional-networks/)

### Execution
```sh
$ python max_pool_parameters.py
```

### License

Apache License, Version 2.0
