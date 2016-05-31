require 'optim'

local folderOfThisFile = (...):match("(.-)[^%.]+$")
require(folderOfThisFile .. 'Trainer')


local TrainerSGD, parent = torch.class('nn.TrainerSGD', 'nn.Trainer')


function TrainerSGD:__init(model)
   self.miniBs = 256
   self.maxIters = 30
   self.noEpochs = 20

   self.optimFunction = optim.sgd
   self.optimParams = {learningRate = 0.1,
                        weightDecay = 1e-4,
                        momentum = 0.9,
                        learningRateDecay = 0.0,
                        nesterov = true,
                        dampening = 0}
   self.model = model
end
