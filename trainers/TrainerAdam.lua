require 'optim'

local folderOfThisFile = (...):match("(.-)[^%.]+$")
require(folderOfThisFile .. 'Trainer')


local TrainerAdam, parent = torch.class('TrainerAdam', 'Trainer')


function TrainerAdam:__init(model)
   self.miniBS = 256
   self.maxIters = 30
   self.noEpochs = 30

   self.optimFunction = optim.adam
   self.optimParams = {learningRate = 0.1}
   self.model = model
end
