-------------------------------------------------------------------------------
-- Trainer
-- * number of iteration, number of epochs, mini-batch size
-- * learning rate, parameters update rule
--
-- Train:
-- * Run mini-batches splits for each epoch.
-- * Update parameters using the "optim" lua package and the predefined feval of the model.
--
-- Test:
-- * Evaluate the loss on test/validation set
-- * Evaluate the adversarial loss for the specific model (net and cost)
-------------------------------------------------------------------------------

require 'nn'


local Trainer = torch.class('nn.Trainer')


---------------
---- Train ----
---------------
function Trainer:train(trainset, validset, testAdversarial, verbose)
   print('Start Training...')

   local dsSize = trainset:size(1)
   local noIters = self:__getNoIters(dsSize)
   local permIdx = torch.randperm(dsSize, 'torch.LongTensor')
   local totaloss = 0
   local accuracy = 0

   local flattenParams = self.model.flattenParams
   local optimParams = self.optimParams
   local optimFunction = self.optimFunction
   local model = self.model
   local miniBs = self.miniBs

   for epoch = 1, self.noEpochs do
      local epochLoss = 0
      local start = os.clock()

      for iter = 1, noIters do
         xlua.progress(iter, noIters)

         -- get mini-batch
         local inputs, labels = trainset:nextBatch(iter, permIdx, miniBs)

         -- get feval for this batch and model
         local feval = model:feval(inputs, labels)

         -- update parameters with self.optimFunction rules
         local _, fs = optimFunction(feval, flattenParams, optimParams)

         -- update loss
         epochLoss = epochLoss + fs[1]
      end

      -- report average error on epoch
      epochLoss = epochLoss / noIters
      totaloss = totaloss + epochLoss
      accuracy = accuracy / noIters

      self:__logging("[Epoch " .. epoch .. "] [Train]          loss: " .. epochLoss, verbose)

      -- validate
      self:test(validset, testAdversarial, verbose)
      print(os.clock() - start)
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
   local dsSize = testset:size(1)
   local noIters = self:__getNoIters(dsSize)
   local permIdx = torch.randperm(dsSize, 'torch.LongTensor')

   local avgLoss = 0
   local avgLossAdv = 0
   local accuracy = 0
   local accuracyAdv = 0

   for iter = 1, noIters do
      -- get mini-batch
      local inputs, labels = testset:nextBatch(iter, permIdx, self.miniBs)
      local iterOutputs, iterLoss = self.model:forward(inputs, labels)
      accuracy = accuracy + getAccuracy(iterOutputs, labels)
      avgLoss = avgLoss + iterLoss

      -- evaluate loss on this mini-batch
      if testAdversarial then
         local inputsAdv = self.model:adversarialSamples(inputs, labels)
         local iterOutputsAdv, iterLossAdv = self.model:forward(inputsAdv, labels)
         accuracyAdv = accuracyAdv + getAccuracy(iterOutputsAdv, labels)
         avgLossAdv = avgLossAdv + iterLossAdv
      end
   end

   -- avg loss
   avgLoss = avgLoss / noIters
   avgLossAdv = avgLossAdv / noIters
   accuracy = accuracy / noIters
   accuracyAdv = accuracyAdv / noIters

   self:__logging("\t[Test] Loss            : " .. avgLoss .. "  Accuracy: " .. accuracy, verbose)

   if testAdversarial then
      self:__logging("\t[Test] Adversarial loss: " .. avgLossAdv .. "  Accuracy: " .. accuracyAdv .. "\n", verbose)
   end
   return avgLoss
end
-----------------
---- END Test ---
-----------------


-----------
-- Utils --
-----------
function getAccuracy(outputs, labels)
   local max_proba, answers_pred = torch.max(outputs, 2)
   local num_correct = torch.eq(answers_pred - labels, 0):sum()
   return num_correct/labels:size(1)
end


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
