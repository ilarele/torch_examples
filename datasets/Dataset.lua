-------------------------------------------------------------------------------
-- Dataset
--
-- Generic dataset class. Each new dataset class should extend it.
-- A Dataset object has 3 DatasetSplits: {trainset, testset, validset}.
-------------------------------------------------------------------------------

require 'paths'

local folderOfThisFile = (...):match("(.-)[^%.]+$")
require(folderOfThisFile .. 'DatasetSplit')


local Dataset = torch.class('Dataset')
local DATA_PATH = 'data/'


---------------
----- Init ----
---------------
function Dataset:__init()
   self.DATA_PATH = 'data/'
   self.TRAIN_VALIDATION_RATIO = 0.8
end


function Dataset:runOnCuda(run)
   local splits = {"trainset", "validset", "testset"}

   if run then
      require 'cunn'
      require 'cudnn'

      for i = 1, 3 do
         self[splits[i]].data = self[splits[i]].data:cuda()
         self[splits[i]].label = self[splits[i]].label:cuda()
      end
   else
      -- float precision is good enough for ML
      for i = 1, 3 do
         self[splits[i]].data = self[splits[i]].data:float()
         self[splits[i]].label = self[splits[i]].label:float()
      end
   end
end
---------------
-- END Init ---
---------------


--------------------
---- Preprocess ----
--------------------
function Dataset:__randomSplitTrainValid(trainAndValidSets, classLabels)
   -- split train in train + validation
   local dsSize = trainAndValidSets.data:size(1)
   local permIdx = torch.randperm(dsSize, 'torch.LongTensor')
   local trainIdx = permIdx[{{1, dsSize * self.TRAIN_VALIDATION_RATIO}}]
   local validIdx = permIdx[{{dsSize * self.TRAIN_VALIDATION_RATIO + 1, dsSize}}]

   local trainset = {}
   trainset.data = trainAndValidSets.data:index(1, trainIdx)
   trainset.label = trainAndValidSets.label:index(1, trainIdx)

   local validset = {}
   validset.data = trainAndValidSets.data:index(1, validIdx)
   validset.label = trainAndValidSets.label:index(1, validIdx)

   trainset = DatasetSplit(trainset, classLabels)
   validset = DatasetSplit(validset, classLabels)
   return trainset, validset
end
------------------------
---- END Preprocess ----
------------------------


--------------------
-- Serialization ---
--------------------
function Dataset:saveMe(objPath)
   print("save obj to path", objPath)
   local newObj = {}
   for key, value in pairs(self) do
      newObj[key] = value
   end
   torch.save(objPath, newObj)
end


function Dataset:loadMe(objPath)
   if paths.filep(objPath) then
      print("load obj from path", objPath)
      local loadedObj = torch.load(objPath)
      for key, value in pairs(loadedObj) do
         self[key] = value
      end
      return true
   end
   return false
end
------------------------
--- END Serialization --
------------------------
