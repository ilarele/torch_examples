require 'optim'
require 'Trainer'

local TrainerAdam, parent = torch.class('nn.TrainerAdam', 'nn.Trainer')


function TrainerAdam:__init(model)
   self.miniBs = 256
   self.maxIters = 30
   self.noEpochs = 30

   self.optimFunction = optim.adam
   self.optimParams = {learningRate = 0.1}
   self.model = model
end
