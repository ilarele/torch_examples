require 'utils'

local DatasetClass, ModelClass, TrainerClass


function custom_setup(arg)
   local cmdOpt = init(arg)
   cmdOpt.runOnCuda = true
   cmdOpt.verbose = true
   cmdOpt.adversarial = true

   -----------------------------------
   -- setup dataset, model and trainer
   -----------------------------------
   require 'TrainerAdam'
   require 'ModelResnetAdversarial'
   require 'DatasetCifarSmall'

   DatasetClass = nn.DatasetCifarSmall
   ModelClass = nn.ModelResnetAdversarial
   TrainerClass = nn.TrainerAdam
   -----------------------------------

   return cmdOpt
end


function main(arg)
   local cmdOpt = custom_setup(arg)

   -- initialize
   local dataset = DatasetClass(cmdOpt.runOnCuda)
   local model = ModelClass(#dataset.classLabels, cmdOpt.runOnCuda)
   local trainer = TrainerClass(model)

   -- train
   local trainLoss = trainer:train(dataset.trainset, dataset.validset, cmdOpt.adversarial, cmdOpt.verbose)
   local testLoss = trainer:test(dataset.testset, cmdOpt.adversarial, cmdOpt.verbose)
end


main(arg)