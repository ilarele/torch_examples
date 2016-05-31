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
require 'image'


local Trainer = torch.class('nn.Trainer')
local FRAME_H = 300
local FRAME_W = 300


---------------
---- Train ----
---------------
function Trainer:train(trainset, validset, printAdversarial, verbose)
   print('Start Training...')

   local dsSize = trainset:size(1)
   local noIters = self:__getNoIters(dsSize)
   local totalLoss = 0
   local accuracy = 0

   local flatParams = self.model.flatParams
   local optimFunction = self.optimFunction
   local optimParams = self.optimParams
   local model = self.model
   local miniBs = self.miniBs

   print("Size params", flatParams:size(1))
   print("Train size", trainset:size(1))
   print("Valid size", validset:size(1))
   print("Batch size", miniBs)
   print("Iterations per epoch", noIters)
   print("Optimization params", self, optimParams)

   for epoch = 1, self.noEpochs do
      local permIdx = torch.randperm(dsSize, 'torch.LongTensor')

      local epochLoss = 0
      for iter = 1, noIters do
         xlua.progress(iter, noIters)

         -- get mini-batch
         local inputs, labels = trainset:nextBatch(iter, permIdx, miniBs)

         -- trick for getting the output out of feval
         local outputs
         local feval = function(x)
            local loss, flatDlossParams
            loss, flatDlossParams, outputs = model:feval(inputs, labels)(x)
            return loss, flatDlossParams
         end

         -- update parameters with self.optimFunction rules
         local _, fs = optimFunction(feval, flatParams, optimParams)

         -- update loss
         epochLoss = epochLoss + fs[1]
         accuracy = accuracy + __getAccuracy(outputs, labels)
      end

      -- report average error on epoch
      epochLoss = epochLoss / noIters
      accuracy = accuracy / noIters
      totalLoss = totalLoss + epochLoss

      __logging("[Epoch " .. epoch .. "] [Train]          loss: " .. epochLoss.. "  Accuracy: " .. accuracy, verbose)

      -- validate
      self:test(validset, printAdversarial, verbose)
   end

   local avgLoss = totalLoss / self.noEpochs
   return avgLoss
end
---------------
-- End Train --
---------------


--------------
---- Test ----
--------------
function Trainer:test(dataset, printAdversarial, verbose)
   local dsSize = dataset:size(1)
   local noIters = self:__getNoIters(dsSize)
   local permIdx = torch.randperm(dsSize, 'torch.LongTensor')

   local avgLoss = 0
   local avgLossAdv = 0
   local accuracy = 0
   local accuracyAdv = 0

   local model = self.model
   local miniBs = self.miniBs

   for iter = 1, noIters do
      -- get mini-batch
      local inputs, labels = dataset:nextBatch(iter, permIdx, miniBs)
      local iterOutputs, iterLoss = model:forward(inputs, labels)
      accuracy = accuracy + __getAccuracy(iterOutputs, labels)
      avgLoss = avgLoss + iterLoss

      -- evaluate loss on this mini-batch
      if printAdversarial then
         local inputsAdv = model:adversarialSamples(inputs, labels)
         local iterOutputsAdv, iterLossAdv = model:forward(inputsAdv, labels)
         accuracyAdv = accuracyAdv + __getAccuracy(iterOutputsAdv, labels)
         avgLossAdv = avgLossAdv + iterLossAdv

         if iter == noIters then
            -- show an adversarial img
            showImageAndAdv(inputs[1], inputsAdv[1], dataset.mean, dataset.stdv)
         end
      end
   end

   -- avg loss
   avgLoss = avgLoss / noIters
   avgLossAdv = avgLossAdv / noIters
   accuracy = accuracy / noIters
   accuracyAdv = accuracyAdv / noIters

   __logging("\t[Test] Loss            : " .. avgLoss .. "  Accuracy: " .. accuracy, verbose)

   if printAdversarial then
      __logging("\t[Test] Adversarial loss: " .. avgLossAdv .. "  Accuracy: " .. accuracyAdv .. "\n", verbose)
   end


   return avgLoss
end
-----------------
---- END Test ---
-----------------


-----------
-- Utils --
-----------
function __getAccuracy(outputs, labels)
   local _, answersPred = torch.max(outputs, 2)
   local numCorrect = torch.eq(answersPred - labels, 0):sum()
   return numCorrect/labels:size(1)
end


function Trainer:__getNoIters(dsSize)
   local noIters = torch.floor(dsSize / self.miniBs)

   if noIters > self.maxIters then
      noIters = self.maxIters
   end

   return noIters
end


function __logging(toPrint, verbose)
   if verbose then
      print(toPrint)
   end
end


function __undoPreprocess(image_tensor, mean, stdv)
   for i = 1, 3 do
      image_tensor[i] = image_tensor[i] * stdv[i]
      image_tensor[i] = image_tensor[i] + mean[i]
   end
   return image_tensor
end


function __genPrintImage(image_tensor, mean, stdv)
   local fullImg = __undoPreprocess(image_tensor, mean, stdv)
   fullImg = fullImg:float()
   local toPrintImg = image.scale(fullImg, FRAME_W, FRAME_H)
   return toPrintImg
end


function showImage(image_tensor, mean, stdv)
   local toPrintImg = __genPrintImage(image_tensor, mean, stdv)
   image.display(toPrintImg)
end


function showImageAndAdv(image_tensor1, image_tensor2, mean, stdv)
   local toPrintImg1 = __genPrintImage(image_tensor1, mean, stdv)
   local toPrintImg2 = __genPrintImage(image_tensor2, mean, stdv)

   local channels = toPrintImg1:size(1)
   local h = toPrintImg1:size(2)
   local w = toPrintImg1:size(3) * 2

   local bothImages = torch.Tensor(channels, h, w)
   bothImages[{{}, {}, {1, w/2}}] = toPrintImg1
   bothImages[{{}, {}, {w/2+1, w}}] = toPrintImg2
   image.display(bothImages)
   -- io.read() -- pause
end
---------------
-- END Utils --
---------------
