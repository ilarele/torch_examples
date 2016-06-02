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
