GOTO samples and run them
- "th sample_name" (th 0.cifar10-basic-cnn.lua)


Details ===

Tutorials
- basic (load dataset, basic train, checkpoints, confusion matrix)
- normal (cuda, autograd)
- advanced (threads, multi-gpu paralel, memory space)


Datasets
- cifar10


Models
- net
- criterion
- add an exact fwd/backward (cost_computing/criterion wrapper)
- generate_adversarial_example


Trainer
- results (multi-metrics)
    - accuracy simple vs advers
    - #change labels for adversarial
    - #change labels (ok to wrong) for adversarial


Train/Test a NN:
dataset = ...
model = ...
trainer(dataset.trainset, dataset.validset)
trainer.show_results(dataset.testset)


