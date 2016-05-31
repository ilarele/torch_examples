require 'utils'


local DatasetClass, ModelClass, TrainerClass


function custom_setup(arg)
   local cmdOpt = init(arg)

   require 'trainers.TrainerSGD'
   require 'models.ModelResnet'
   require 'datasets.DatasetCifarSmall'

   DatasetClass = nn.DatasetCifarSmall
   ModelClass = nn.ModelResnet
   TrainerClass = nn.TrainerSGD

   return cmdOpt
end


function main(arg)
   torch.manualSeed(1)
   local cmdOpt = custom_setup(arg)

   -- initialize
   local dataset = DatasetClass(cmdOpt.runOnCuda)
   local model = ModelClass(#dataset.classLabels, cmdOpt.runOnCuda)
   local trainer = TrainerClass(model)

   -- train
   local trainLoss = trainer:train(dataset.trainset, dataset.validset, cmdOpt.printAdversarial, cmdOpt.verbose)
   local testLoss = trainer:test(dataset.testset, cmdOpt.printAdversarial, cmdOpt.verbose)

   print("Final Train Loss: ", trainLoss)
   print("Final Test Loss : ", testLoss)
end


main(arg)