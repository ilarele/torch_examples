require 'optim'
require 'Trainer'

local TrainerAdam, parent = torch.class('nn.TrainerAdam', 'nn.Trainer')


function TrainerAdam:__init(model)
   self.miniBs = 128
   self.maxIters = 50
   self.noEpochs = 100

   self.optimFunction = optim.adam
   self.optimParams = {learningRate = 0.1}
   self.model = model
end
