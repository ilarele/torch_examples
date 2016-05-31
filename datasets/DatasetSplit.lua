-------------------------------------------------------------------------------
-- DatasetSplit
--
-- This class refers to trainset, validset and testset splits.
-- Implements: __index[], size() and nextBatch()
-- Contains mean and stdv tensors, used in normalization.
-------------------------------------------------------------------------------

local DatasetSplit = torch.class('nn.DatasetSplit')


function DatasetSplit:__init(dict)
   assert(dict)
   self.data = dict.data
   self.label = dict.label
end


function DatasetSplit:nextBatch(iter, permutation, batchSize)
   -- Returns a batch from the shuffled dataset
   --    permutation: random permutation indexes
   --    startIndex: it refers to the permutation

   local datasetSize = self:size(1)
   assert(datasetSize == permutation:size(1))

   local startIndex = batchSize * (iter - 1) + 1

   local crtBatchSize = math.min(batchSize, datasetSize - startIndex + 1)
   local endIndex = crtBatchSize + startIndex - 1

   local shuffledIdx = permutation[{{startIndex, endIndex}}]
   local batchIdx = permutation:index(1, shuffledIdx)

   local inputs = self.data:index(1, batchIdx)
   local labels = self.label:index(1, batchIdx)

   return inputs, labels
end


function DatasetSplit:__normalize(mean, stdv)
   assert(mean, "Failed to normalize dataset, mean is nil")
   assert(stdv, "Failed to normalize dataset, stdv is nil")

   -- over each image channel (np.vectorize this)
   for i = 1, 3 do
      self.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
      self.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
   end
end


function DatasetSplit:__convertToFloat()
   self.data = self.data:float()
   self.label = self.label:long()
end


function DatasetSplit:__getMeanStdv()
   local mean = {}
   local stdv = {}

   for i = 1, 3 do
      mean[i] = self.data[{ {}, {i}, {}, {}  }]:mean()
      stdv[i] = self.data[{ {}, {i}, {}, {}  }]:std()
   end

   return mean, stdv
end


function DatasetSplit:preprocess(mean, stdv)
   self:__convertToFloat()
   if not mean or not stdv then
      mean, stdv = self:__getMeanStdv()
   end

   self.mean = mean
   self.stdv = stdv
   self:__normalize(mean, stdv)
   return mean, stdv
end


-----------------------------
-- torch dataset interface --
-----------------------------
function DatasetSplit:__index__(i)
   if type(i) == 'number' then
      return {self.data[i], self.label[i]}
   end
   return rawget(self, key)
end


function DatasetSplit:size()
   return self.data:size(1)
end
-------------------------------------
---- END torch dataset interface ----
-------------------------------------
