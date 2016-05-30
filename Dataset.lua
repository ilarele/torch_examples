--
-- A dataset
-- * has 3 splits: dataset.trainset, dataset.testset, dataset.validset

require 'paths'
require 'DatasetSplit'


local Dataset = torch.class('nn.Dataset')
local DATA_PATH = 'data/'


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


-- -----------------------------
-- -- torch dataset interface --
-- -----------------------------
-- function Dataset:__comply_to_interface(dataset_split)
--    -- make it indexable []
--    setmetatable(dataset_split,
--       {__index = function(t, i)
--                   return {t.data[i], t.label[i]}
--                end}
--    );

--    -- add size()
--    function dataset_split:size()
--       return self.data:size(1)
--    end
-- end
-- -------------------------------------
-- ---- END torch dataset interface ----
-- -------------------------------------


--------------------
---- Preprocess ----
--------------------
function Dataset:__randomSplitTrainValid(trainAndValidSets)
   -- split train in train + validation
   torch.manualSeed(1)
   local dsSize = trainAndValidSets.data:size(1)
   local permIdx = torch.randperm(dsSize, 'torch.LongTensor')
   local trainIdx = permIdx[{{1, dsSize * self.TRAIN_VALIDATION_RATIO}}]
   local validIdx = permIdx[{{dsSize * self.TRAIN_VALIDATION_RATIO + 1, dsSize}}]

   -- trainset = trainAndValidSets
   local trainset = {}
   trainset.data = trainAndValidSets.data:index(1, trainIdx)
   trainset.label = trainAndValidSets.label:index(1, trainIdx)

   local validset = {}
   validset.data = trainAndValidSets.data:index(1, validIdx)
   validset.label = trainAndValidSets.label:index(1, validIdx)


   trainset = nn.DatasetSplit(trainset)
   validset = nn.DatasetSplit(validset)
   return trainset, validset
end

------------------------
---- END Preprocess ----
------------------------


--------------------
-- serialization ---
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
--- END serialization --
------------------------
