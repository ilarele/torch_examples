require 'optim'
require 'Trainer'

local TrainerSGD, parent = torch.class('nn.TrainerSGD', 'nn.Trainer')


function TrainerSGD:__init(model)
   self.miniBs = 128
   self.maxIters = 50
   self.noEpochs = 30

   self.optimFunction = optim.sgd
   self.optimParams = {learningRate=0.05,
                        weightDecay=1e-4,
                        momentum=0.9,
                        learningRateDecay = 0.0,
                        nesterov = true,
                        dampening = 0}
   self.model = model
end


