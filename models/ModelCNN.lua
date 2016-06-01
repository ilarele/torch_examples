-------------------------------------------------------------------------------
-- ModelCNN
-- toy example
-------------------------------------------------------------------------------

local folderOfThisFile = (...):match("(.-)[^%.]+$")
require(folderOfThisFile .. 'Model')


local ModelCNN, parent = torch.class('ModelCNN', 'Model')


function ModelCNN:__init(noClassLabels, optRunOnCuda)
   local opt = {}
   opt.noClassLabels = noClassLabels

   parent.__init(self, optRunOnCuda, opt)
end


function ModelCNN:__createModel(opt)
   local net = nn.Sequential()

   -- initialize parameters
   net = nn.Sequential()

   net:add(nn.SpatialConvolution(3, 10, 5, 5))
   net:add(nn.SpatialBatchNormalization(10, 1e-3))
   net:add(nn.ReLU())
   net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   net:add(nn.SpatialConvolution(10, 30, 5, 5))
   net:add(nn.SpatialBatchNormalization(30, 1e-3))
   net:add(nn.ReLU())
   net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   net:add(nn.View(30*5*5))
   net:add(nn.Linear(30*5*5, 120))
   net:add(nn.Linear(120, 84))
   net:add(nn.ReLU())
   net:add(nn.Linear(84, opt.noClassLabels))
   net:add(nn.LogSoftMax())

   local params, _ = net:getParameters()
   params:uniform(-0.1, 0.1)

   return net, nn.ClassNLLCriterion()
end
