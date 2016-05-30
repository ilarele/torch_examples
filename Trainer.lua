require 'nn'

local Trainer = torch.class('nn.Trainer')


---------------
---- Train ----
---------------
function Trainer:train(trainset, validset, testAdversarial, verbose)
   print('[Training...]')

   local dsSize = trainset:size(1)
   local noIters = self:__getNoIters(dsSize)
   local permIdx = torch.randperm(dsSize, 'torch.LongTensor')
   local model = self.model
   local totaloss = 0

   print(trainset)
   for epoch = 1, self.noEpochs do
      local epochLoss = 0
      for iter = 1, noIters do
         -- get mini-batch
         local inputs, labels = trainset:nextBatch(iter, permIdx, self.miniBs)

         -- get feval for this batch and model
         local feval = model:feval(inputs, labels)

         -- update params with self.optimFunction rules
         local _, fs, _ = self.optimFunction(feval, model.flattenParams, self.optimParams)

         -- update loss
         epochLoss = epochLoss + fs[1]
      end

      -- report average error on epoch
      epochLoss = epochLoss / noIters
      totaloss = totaloss + epochLoss
      self:__logging(epoch .. " train_loss   " .. epochLoss, verbose)
      self:test(validset, testAdversarial, verbose)
   end

   local avgLoss = totaloss / self.noEpochs
   return avgLoss
end
---------------
-- End Train --
---------------


--------------
---- Test ----
--------------
function Trainer:test(testset, testAdversarial, verbose)
   -- print('[Testing...]')

   local dsSize = testset:size(1)
   local noIters = self:__getNoIters(dsSize)
   local permIdx = torch.randperm(dsSize, 'torch.LongTensor')

   local avgLoss = 0
   local avgLossAdv = 0
   for iter = 1, noIters do
      -- get mini-batch
      local inputs, labels = testset:nextBatch(iter, permIdx, self.miniBs)
      local _, iterLoss = self.model:forward(inputs, labels)

      -- evaluate loss on this mini-batch
      if testAdversarial then
         local inputsAdv = self.model:adversarialSamples(inputs, labels)
         local _, iterLossAdv = self.model:forward(inputsAdv, labels)
         avgLossAdv = avgLossAdv + iterLossAdv
      end

      -- update loss
      avgLoss = avgLoss + iterLoss
   end

   -- avg loss
   avgLoss = avgLoss / noIters
   avgLossAdv = avgLossAdv / noIters

   self:__logging("test_loss    " .. avgLoss, verbose)
   self:__logging("test_loss_adv " .. avgLossAdv, verbose)
   return avgLoss
end
-----------------
---- END Test ---
-----------------


-----------
-- Utils --
-----------
function Trainer:__getNoIters(dsSize)
   local noIters = torch.floor(dsSize / self.miniBs)
   if noIters > self.maxIters then
      noIters = self.maxIters
   end
   return noIters
end


function Trainer:__logging(toPrint, verbose)
   if verbose then
      print(toPrint)
   end
end
---------------
-- END Utils --
---------------
