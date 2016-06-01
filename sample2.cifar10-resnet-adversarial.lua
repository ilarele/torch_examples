require 'utils'


local DatasetClass, ModelClass, TrainerClass


function custom_setup(arg)
   local cmdOpt = init(arg)

   require 'trainers.TrainerSGD'
   require 'models.ModelResnetAdversarial'
   require 'datasets.DatasetCifarSmall'

   DatasetClass = DatasetCifarSmall
   ModelClass = ModelResnetAdversarial
   TrainerClass = TrainerSGD

   return cmdOpt
end


function main(arg)
   local cmdOpt = custom_setup(arg)

   -- initialize
   local dataset = DatasetClass(cmdOpt.runOnCuda)
   local model = ModelClass(#dataset.classLabels, cmdOpt.runOnCuda)
   local trainer = TrainerClass(model)

   -- train
   local trainLoss = trainer:train(dataset.trainset, dataset.validset, cmdOpt)
   local testLoss = trainer:test(dataset.testset, cmdOpt)

   print("Done")
end


main(arg)
