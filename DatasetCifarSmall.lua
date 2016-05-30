--------------------
-- Cifar10 dataset
--------------------


require 'paths'
require 'Dataset'


local DatasetCifarSmall, parent = torch.class('nn.DatasetCifarSmall', 'nn.Dataset')


---------------
----- Init ----
---------------
function DatasetCifarSmall:__init(optRunOnCuda)
   parent.__init(self)
   self:__cifar10()
   self:runOnCuda(optRunOnCuda)
end


function DatasetCifarSmall:__cifar10()
   print('Loading dataset...')

   local DATASET_NAME = 'cifar10'
   local DATASETPATH = string.format('%s%s/', self.DATA_PATH, DATASET_NAME)
   local PREPROC_DATASETPATH = DATASETPATH .. 'cifar10-preproc.t7'

   local loaded = self:loadMe(PREPROC_DATASETPATH)
   if loaded then
      return
   end

   local trainsetPath = string.format('%scifar10-train.t7', DATASETPATH)
   local testsetPath = string.format('%scifar10-test.t7', DATASETPATH)
   local classLabels = {'airplane', 'automobile', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

   local mean, stdv

   self:__download_cifar10(DATASETPATH)
   local allSplits = self:__rawCifar10Splits(trainsetPath, testsetPath)

   for splitIdx = 1, 3 do
      local split = allSplits[splitIdx]
      mean, stdv = split:preprocess(mean, stdv)
   end

   self.trainset = allSplits[1]
   self.validset = allSplits[2]
   self.testset = allSplits[3]
   self.classLabels = classLabels

   self:saveMe(PREPROC_DATASETPATH)
end
---------------
-- END Init ---
---------------


--------------------
----- Download -----
--------------------
function DatasetCifarSmall:__download_cifar10(datasetPath)
   local result = false
   local datasetExists = paths.dirp(datasetPath)
   if not datasetExists then
      os.execute('git clone https://github.com/soumith/cifar.torch ; cd cifar.torch ; th Cifar10BinToTensor.lua')

      os.execute('mkdir -p data/cifar10/')
      os.execute('mv cifar.torch/cifar10-train.t7 data/cifar10/')
      os.execute('mv cifar.torch/cifar10-test.t7 data/cifar10/')

      result = true

      datasetExists = paths.dirp(datasetPath)
      if not datasetExists then
         print('Error occured when downloading dataset. See download_cifar10 function')
         self:deleteCifar10(datasetPath)
      else
         self:deleteCifar10("cifar.torch")
      end
   end

   return result
end


function DatasetCifarSmall:__deleteCifar10(datasetPath)
   os.execute('rm -rf ' .. datasetPath)
end
--------------------
-- END Download ----
--------------------


------------------------
-- Preprocess - split --
------------------------
function DatasetCifarSmall:__rawCifar10Splits(trainsetPath, testsetPath)
   --[[
   Load torch datasets. Increment class labels (in torch e indexare de la 1).
   ]]--
   local trainAndValidSets = torch.load(trainsetPath)
   local testset = torch.load(testsetPath)

   -- fix labels
   trainAndValidSets.label = trainAndValidSets.label + 1
   testset.label = testset.label + 1

   local trainset, validset = self:__randomSplitTrainValid(trainAndValidSets)
   testset = nn.DatasetSplit(testset)
   return {trainset, validset, testset}
end
----------------------------
-- END Preprocess - split --
----------------------------

