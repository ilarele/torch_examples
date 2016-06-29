This repository contains samples for generating adversarial examples in torch, using autograd (https://github.com/twitter/torch-autograd) for Automatic Differentiation.


###Run sample (Train/Test NN)###
* `th sample_name.lua -runOnCuda`
* `th sample_name.lua -runOnCuda -printAdversarial`
* `qlua sample_name.lua -runOnCuda -printAdversarial -showImages`
* `-printAdversarial` option takes more time (in order to compute the network outputs for the adversarial examples)


##Dataset##
A dataset object has 3 DatasetSplits
* `{trainset, testset, validset}`
* number of classes (for classification tasks)


##DatasetSplits##
* mean, stdv, classLabels for the dataset

##Model##
A model contains the net architecture, the criterion and the feval function (forward and backward steps). It also contains code for generating adversarial examples.
* `net`
* `criterion`
* `feval()`, `forward()`, `backward()`
* `adversarialSamples()`

###ModelResnetAdversarial###
* trained with adversarial cost
* evaluated on its own adversarial examples (w.r.t. its own adversarial cost)


##Trainer##
A trainer contains details about:
* number of iteration, number of epochs, mini-batch size
* parameters update algorithm (function and parameters: ex.sgd with `learningRate` 0.1)
* printing out metrics about the training (accuracy, accuracy on adversarial examples)
* plotting adversarial images


###Help###
* `th sample_name.lua -h`


##Adversarial examples for ResNet20 on CIFAR10
General setup:
- simple training (not fine tuned)
- CIFAR10 raw data (preprocess: normalization only)
- grad_cost_x = dcost(x)/dx
- adversarial_x(x) = x + grad_cost_x/norm(grad_cost_x)

1. Training with normal cost: ClassNLLCriterion(x)
2. Training with adversarial cost: (ClassNLLCriterion(x) + ClassNLLCriterion(adversarial_x))/2


| Metric              \             Cost type |  Normal Cost | Adversarial Cost |
| ------------------------------------------- |:------------:|:----------------:|
| Accuracy                                    |      73%     |        68%       |
| Accuracy on adversarial examples            |      29%     |        57%       |
| % Adversarial samples with changed label    |      53%     |        34%       |
| Visually     | ![Normal Cost](https://raw.githubusercontent.com/ilarele/torch_examples/master/images/simple1.jpg)|  ![Adversarial Cost](https://raw.githubusercontent.com/ilarele/torch_examples/master/images/adversarial1.jpg)|
| Visually     | ![Normal Cost](https://raw.githubusercontent.com/ilarele/torch_examples/master/images/simple2.jpg)|  ![Adversarial Cost](https://raw.githubusercontent.com/ilarele/torch_examples/master/images/adversarial2.jpg)|
| Visually     | ![Normal Cost](https://raw.githubusercontent.com/ilarele/torch_examples/master/images/simple3.jpg)|  ![Adversarial Cost](https://raw.githubusercontent.com/ilarele/torch_examples/master/images/adversarial3.jpg)|

