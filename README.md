This repository contains samples for generating adversarial examples in torch, using autograd (https://github.com/twitter/torch-autograd) for Automatic Differentiation.

##Dataset##
A dataset object has 3 DatasetSplits
* {trainset, testset, validset}
* number of classes (for classification tasks)


##DatasetSplits##
* mean, stdv of the dataset


##Model##
A model contains the net architecture, the criterion and the feval function (forward and backward steps). It also contains code for generating adversarial examples.
* net
* criterion
* feval(), forward(), backward()
* adversarialSamples()


##Trainer##
A trainer contains details about:
* number of iteration, number of epochs, mini-batch size
* parameters update algorithm (function and parameters: ex.sgd with learningRate=0.1)
* printing out metrics about the training (accuracy, accuracy on adversarial examples)
* plotting adversarial images


##Train/Test a NN##
local dataset = DatasetClass(cmdOpt.runOnCuda)
local model = ModelClass(#dataset.classLabels, cmdOpt.runOnCuda)
local trainer = TrainerClass(model)

local trainLoss = trainer:train(dataset.trainset, dataset.validset, cmdOpt.printAdversarial, cmdOpt.verbose)
local testLoss = trainer:test(dataset.testset, cmdOpt.printAdversarial, cmdOpt.verbose)

###Run sample###
* th sample.lua -runOnCuda
* -printAdversarial option takes more time (in order to compute the network outputs for the adversarial examples)

###Help###
* th sample_name.lua -h
