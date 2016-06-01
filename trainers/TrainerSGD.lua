require 'optim'

local folderOfThisFile = (...):match("(.-)[^%.]+$")
require(folderOfThisFile .. 'Trainer')


local TrainerSGD, parent = torch.class('TrainerSGD', 'Trainer')


function TrainerSGD:__init(model)
   self.miniBS = 256
   self.maxIters = 20
   self.noEpochs = 150

   self.optimFunction = optim.sgd
   self.optimParams = {learningRate = 0.1,
                        weightDecay = 1e-4,
                        momentum = 0.9,
                        learningRateDecay = 0.0,
                        nesterov = true,
                        dampening = 0}
   self.model = model
end


function Trainer:updateOptimParams(epoch)
   local decay = 0
   if epoch >= 122 then
      decay = 2
   elseif epoch >= 81 then
      decay = 1
   else
      decay = 0
   end

   local lr = self.optimParams.learningRate
   lr = lr * math.pow(0.1, decay)
   self.optimParams.learningRate = lr
end