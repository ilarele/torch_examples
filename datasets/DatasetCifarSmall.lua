-------------------------------------------------------------------------------
-- DatasetCifarSmall
--
-- Cifar10 dataset loader. Download, preprocess, split into DatasetSplits
-- {trainset, testset, validset} and save on disk.
-- Fast loading is enabled by default. Loading from previously saved
-- preprocessed Dataset.
-------------------------------------------------------------------------------

require 'paths'

local folderOfThisFile = (...):match("(.-)[^%.]+$")
require(folderOfThisFile .. 'Dataset')


local DatasetCifarSmall, parent = torch.class('DatasetCifarSmall', 'Dataset')

local TRAINSET_NAME = 'cifar10-train.t7'
local TESTSET_NAME = 'cifar10-test.t7'
local DATASET_NAME = 'cifar10'
local DATASET_PREPROC_NAME = 'cifar10-preproc.t7'

local DATASETPATH, TRAINSET_PATH, TESTSET_PATH, PREPROC_DATASETPATH

local CLASS_LABELS = {'airplane', 'automobile', 'bird', 'cat',
         'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}


---------------
----- Init ----
---------------
function DatasetCifarSmall:__init(optRunOnCuda)
   parent.__init(self)

   -- build paths
   DATASETPATH = string.format('%s%s/', self.DATA_PATH, DATASET_NAME)
   TRAINSET_PATH = string.format('%s%s', DATASETPATH, TRAINSET_NAME)
   TESTSET_PATH = string.format('%s%s', DATASETPATH, TESTSET_NAME)
   PREPROC_DATASETPATH = DATASETPATH .. DATASET_PREPROC_NAME

   self:__cifar10()
   self:runOnCuda(optRunOnCuda)
end


function DatasetCifarSmall:__cifar10()
   print('Loading dataset...')


   local loaded = self:loadMe(PREPROC_DATASETPATH)
   if loaded then
      return
   end

   self:__download_cifar10(DATASETPATH, TRAINSET_NAME, TESTSET_NAME)

   local mean, stdv

   local allSplits = self:__rawCifar10Splits(TRAINSET_PATH, TESTSET_PATH)

   for splitIdx = 1, 3 do
      local split = allSplits[splitIdx]
      mean, stdv = split:preprocess(mean, stdv)
   end

   self.trainset = allSplits[1]
   self.validset = allSplits[2]
   self.testset = allSplits[3]
   self.classLabels = CLASS_LABELS

   self:saveMe(PREPROC_DATASETPATH)
end
---------------
-- END Init ---
---------------


--------------------
----- Download -----
--------------------
-- TODO: fix this mess
function DatasetCifarSmall:__download_cifar10(datasetPath, trainName, testName)
   local downloadedDir = 'cifar.torch/'

   local downloadedDirExists = paths.dirp(downloadedDir)
   if not downloadedDirExists then
      print "Downloading dataset"
      os.execute('git clone https://github.com/soumith/cifar.torch ; cd cifar.torch ; th Cifar10BinToTensor.lua')
   end
   local datasetExists = paths.dirp(datasetPath)
   if not datasetExists then
      print("Extracting dataset")
      os.execute('mkdir -p ' .. datasetPath)
      os.execute('cp ' .. downloadedDir .. trainName .. ' ' .. datasetPath)
      os.execute('cp ' .. downloadedDir .. testName .. ' ' .. datasetPath)
   end
   local trainsetExists = paths.filep(datasetPath .. trainName)
   local testsetExists = paths.filep(datasetPath .. testName)
   if not trainsetExists or not testsetExists then
      self:__rmforce(datasetPath)
   else
      self:__rmforce(downloadedDir)
   end
end


function DatasetCifarSmall:__rmforce(datasetPath)
   os.execute('rm -rf ' .. datasetPath)
end
--------------------
-- END Download ----
--------------------


------------------------
-- Preprocess - Split --
------------------------
function DatasetCifarSmall:__rawCifar10Splits(TRAINSET_PATH, TESTSET_PATH)
   local trainAndValidSets = torch.load(TRAINSET_PATH)
   local testset = torch.load(TESTSET_PATH)

   -- fix labels
   trainAndValidSets.label = trainAndValidSets.label + 1
   testset.label = testset.label + 1

   local trainset, validset = self:__randomSplitTrainValid(trainAndValidSets, CLASS_LABELS)
   testset = DatasetSplit(testset, CLASS_LABELS)
   return {trainset, validset, testset}
end
----------------------------
-- END Preprocess - Split --
----------------------------

