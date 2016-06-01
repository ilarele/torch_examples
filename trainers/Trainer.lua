-------------------------------------------------------------------------------
-- Trainer
-- * number of iterations, number of epochs, mini-batch size
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

require 'image'


local Trainer = torch.class('Trainer')


---------------
---- Train ----
---------------
function Trainer:train(trainset, validset, opt)
   print('Start Training...')

   local dsSize = trainset:size(1)
   local noIters = self:__getNoIters(dsSize)
   local totalLoss = 0
   local accuracy = 0

   local flatParams = self.model.flatParams
   local optimFunction = self.optimFunction
   local model = self.model
   local miniBS = self.miniBS

   print("Size params", flatParams:size(1))
   print("Train size", trainset:size(1))
   print("Valid size", validset:size(1))
   print("Batch size", miniBS)
   print("Iterations per epoch", noIters)
   print("Optimization params", self, self.optimParams)

   for epoch = 1, self.noEpochs do
      local permIdx = torch.randperm(dsSize, 'torch.LongTensor')
      self:updateOptimParams(epoch)

      local epochLoss = 0
      for iter = 1, noIters do
         xlua.progress(iter, noIters)

         -- get mini-batch
         local inputs, labels = trainset:nextBatch(iter, permIdx, miniBS)

         -- trick for getting the output out of feval
         local outputs
         local feval = function(x)
            local loss, flatDlossParams
            loss, flatDlossParams, outputs = model:feval(inputs, labels)(x)
            return loss, flatDlossParams
         end

         -- update parameters with self.optimFunction rules
         local _, fs = optimFunction(feval, flatParams, self.optimParams)

         -- update loss
         epochLoss = epochLoss + fs[1]
         accuracy = accuracy + __getAccuracy(outputs, labels)
      end

      -- report average error on epoch
      epochLoss = epochLoss / noIters
      accuracy = accuracy / noIters
      totalLoss = totalLoss + epochLoss

      -- logging
      __logging("[Epoch " .. epoch .. "] [Train]          loss: " .. epochLoss.. "  Accuracy: " .. accuracy, opt.verbose)

      -- validate
      self:test(validset, opt)
   end

   local avgLoss = totalLoss / self.noEpochs
   return avgLoss
end


function Trainer:updateOptimParams(epoch)
end
---------------
-- End Train --
---------------


--------------
---- Test ----
--------------
function Trainer:test(dataset, opt)
   local dsSize = dataset:size(1)
   local noIters = self:__getNoIters(dsSize)
   local permIdx = torch.randperm(dsSize, 'torch.LongTensor')

   local avgLoss = 0
   local avgLossAdv = 0
   local accuracy = 0
   local accuracyAdv = 0

   local model = self.model
   local miniBS = self.miniBS

   for iter = 1, noIters do
      -- get mini-batch
      local inputs, labels = dataset:nextBatch(iter, permIdx, miniBS)
      local iterOutputs, iterLoss = model:forward(inputs, labels)
      accuracy = accuracy + __getAccuracy(iterOutputs, labels)
      avgLoss = avgLoss + iterLoss

      -- evaluate loss on this mini-batch
      if opt.printAdversarial then
         if opt.showImages then
            iterOutputs = iterOutputs:clone()
         end

         local inputsAdv = model:adversarialSamples(inputs, labels)
         local iterOutputsAdv, iterLossAdv = model:forward(inputsAdv, labels)
         accuracyAdv = accuracyAdv + __getAccuracy(iterOutputsAdv, labels)
         avgLossAdv = avgLossAdv + iterLossAdv

         if opt.showImages then
            if iter == noIters then
               -- show an adversarial img
               showAdversarialImage(inputs, inputsAdv, labels, iterOutputs, iterOutputsAdv, dataset)
            end
         end
      end
   end

   -- avg loss
   avgLoss = avgLoss / noIters
   avgLossAdv = avgLossAdv / noIters
   accuracy = accuracy / noIters
   accuracyAdv = accuracyAdv / noIters

   -- logging
   __logging("\t[Test] Loss            : " .. avgLoss .. "  Accuracy: " .. accuracy, opt.verbose)

   if opt.printAdversarial then
      __logging("\t[Test] Adversarial loss: " .. avgLossAdv .. "  Accuracy: " .. accuracyAdv .. "\n", opt.verbose)
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
   local noIters = torch.floor(dsSize / self.miniBS)

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
----------------
-- END Utils ---
----------------


------------
-- Images --
------------
local FRAME_H = 200
local FRAME_W = 200


function __undoPreprocess(imageTensor, mean, stdv)
   for i = 1, 3 do
      imageTensor[i] = imageTensor[i] * stdv[i]
      imageTensor[i] = imageTensor[i] + mean[i]
   end
   return imageTensor
end


function __genPrintImage(imageTensor, mean, stdv)
   local fullImg = __undoPreprocess(imageTensor, mean, stdv)
   fullImg = fullImg:float()
   local toPrintImg = image.scale(fullImg, FRAME_W, FRAME_H)
   return toPrintImg
end


function showImage(imageTensor, mean, stdv)
   local toPrintImg = __genPrintImage(imageTensor, mean, stdv)
   image.display(toPrintImg)
end
----------------
-- END Images --
----------------


----------------------------
---- Adversarial Images ----
----------------------------
function showNormalAndAdv(imageTensor1, imageTensor2, predY1, predY2, dataset)
   local classLabels = dataset.classLabels
   local mean, stdv = dataset.mean, dataset.stdv

   local toPrintImg1 = __genPrintImage(imageTensor1, mean, stdv)
   local toPrintImg2 = __genPrintImage(imageTensor2, mean, stdv)
   local predY1Cls = classLabels[predY1]
   local predY2Cls = classLabels[predY2]

   local channels = toPrintImg1:size(1)
   local h = toPrintImg1:size(2)
   local w = toPrintImg1:size(3) * 2

   -- glue together the 2 images
   local bothImages = torch.Tensor(channels, h, w)
   bothImages[{{}, {}, {1, w / 2}}] = toPrintImg1
   bothImages[{{}, {}, {w / 2 + 1, w}}] = toPrintImg2

   -- draw
   local drawOpt = {inplace = true, size = 2}
   local text = predY1Cls .. " vs " .. predY2Cls
   image.drawText(bothImages, text, 40, 10, drawOpt)
   image.display(bothImages)
   -- io.read() -- pause
end


function __getAdversarialWrong(labels, iterOutputs, iterOutputsAdv)
   local _, preds = torch.max(iterOutputs, 2)
   local _, predsAdv = torch.max(iterOutputsAdv, 2)

   local normalOk = torch.eq(preds - labels, 0)
   local advWrong = torch.ne(labels - predsAdv, 0)
   local result = normalOk:cmul(advWrong)
   local indexesOk = torch.nonzero(result:float())

   local returnIdx = indexesOk[1][1]
   return returnIdx, preds[returnIdx][1], predsAdv[returnIdx][1]
end

function showAdversarialImage(inputs, inputsAdv, labels, iterOutputs, iterOutputsAdv, dataset)
   local wrongIdx, predY1, predY2 = __getAdversarialWrong(labels, iterOutputs, iterOutputsAdv)

   showNormalAndAdv(inputs[wrongIdx], inputsAdv[wrongIdx], predY1, predY2, dataset)
end
----------------------------
-- END Adversarial Images --
----------------------------

