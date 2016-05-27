require 'optim'
require 'Trainer'

local TrainerAdam, parent = torch.class('nn.TrainerAdam', 'nn.Trainer')


function TrainerAdam:__init(model)
    self.learn_rate = 0.1
    self.mini_bs = 128
    self.max_iters = 50
    self.no_epochs = 10

    self.optim_function = optim.adam
    self.optim_params = {learningRate = self.learn_rate}
    self.model = model
end


