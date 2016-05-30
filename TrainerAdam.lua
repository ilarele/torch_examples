require 'optim'
require 'Trainer'

local TrainerAdam, parent = torch.class('nn.TrainerAdam', 'nn.Trainer')


function TrainerAdam:__init(model)
   self.learningRate = 0.1
   self.miniBs = 128
   self.maxIters = 50
   self.noEpochs = 10

   self.optimFunction = optim.adam
   self.optimParams = {learningRate = self.learningRate}
   self.model = model
end


