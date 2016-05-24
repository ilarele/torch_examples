require 'optim'
require 'Trainer'

local TrainerAdam, parent = torch.class('nn.TrainerAdam', 'nn.Trainer')


function TrainerAdam:__init(model)
    parent.__init(self)

    self.learn_rate = 0.1
    self.mini_bs = 128
    self.max_iters = 50
    self.no_epochs = 10

    model.optim_function = optim.sgd
    model.optim_params = {learningRate = self.learn_rate}
    self.model = model
end


function TrainerAdam:__complete_results(trainset, validset)
end

