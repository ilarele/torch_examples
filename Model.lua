-------------------------------------------------------------------------------
-- Model
-- It mainly contains the net, the criterion and the feval function
-- (forward and backward steps).
--
-- The Model contains:
-- * net - neural net architecture
-- * criterion - evaluation criterion, cost
-- * flattenParams
-- * flattenDlossDparams
--
-- Used for adversarial examples:
-- * adModelForward - autograd (automatic differentiate) version of net:forward
-- * adParams - autograd version of the net parameters
-- * adCriterionForward - autograd version of criterion:forward
--
-- Adversarial Examples
-- * Simple adversarial examples are computed by adding to the Input the
-- gradient of the Cost w.r. Input.
-- * Default Cost is supposed to be the criterion applied (only) on input.
-- * For a more advanced adversarial cost, see ModelResnetAdversarial.lua
-------------------------------------------------------------------------------

local nn = require 'nn'
local autograd = require 'autograd'
local Model = torch.class('nn.Model')


----------------
-- Init Model --
----------------
function Model:__init(optRunOnCuda, opt)
   local net, criterion = self:__createModel(opt)

   -- init
   self.net = net
   self.criterion = criterion
   self:runOnCuda(optRunOnCuda)
   self.flattenParams, self.flattenDlossDparams = self.net:getParameters()

   -- autograd (automatic differentiate) transformations
   -- used for adversarial examples
   self.adModelForward, self.adParams = autograd.functionalize(self.net)
   self.adCriterionForward = autograd.functionalize(self.criterion)
end


function Model:runOnCuda(run)
   -- if changes are made, net parameters pointers are changed.
   -- so update flatten params.
   if run then
      self.net:cuda()
      self.criterion:cuda()
   else
      self.net:float()
      self.criterion:float()
   end
end


function Model:__createModel(opt)
   assert(false, "__createModel should be implemented in each model")
   return nil
end
--------------
-- END Init --
--------------


--------------------
-- Serialization ---
--------------------
function Model:saveMe(obj_path)
   print("save obj to path", obj_path)
   local newObj = {}
   newObj.net = self.net
   newObj.criterion = self.criterion

   torch.save(objPath, newObj)
end


function Model:loadMe(objPath)
   if paths.filep(objPath) then
      print("load obj from path", objPath)

      local loadObj = torch.load(objPath)
      self.net = loadObj.net
      self.criterion = loadObj.criterion

      return true
   end
   return false
end
------------------------
--- END Serialization --
------------------------


---------------------------------
------------- Feval -------------
---------------------------------
function Model:feval(inputs, labels)
   -- forward the input
   -- back-propagate the loss
   -- return loss, params, dCost/dX (aka gradInput)
   return function(x)
      if x ~= self.flattenParams then
         self.flattenParams:copy(x)
      end
      self.flattenDlossDparams:zero()

      local outputs, loss = self:forward(inputs, labels)
      local _, gradInput = self:backward(inputs, outputs, labels)

      return loss, self.flattenDlossDparams
   end
end


function Model:forward(inputs, labels)
   -- compute the loss
   local outputs = self.net:forward(inputs)
   local loss = self.criterion:forward(outputs, labels)

   return outputs, loss
end


function Model:backward(inputs, outputs, labels)
   -- compute the loss
   local dloss_dout = self.criterion:backward(outputs, labels)
   local gradInput = self.net:backward(inputs, dloss_dout)

   return dloss_dout, gradInput
end
---------------------------------
----------- END Feval -----------
---------------------------------


--------------------------------
------ Adversarial examples ----
--------------------------------
local EPS = 50
local adModelForward, adCriterionForward

function adCost(x, adParams, y)
   local yPred = adModelForward(adParams, x)
   local loss = adCriterionForward(yPred, y)
   return loss
end


function Model:adversarialSamples(x, y)
   adModelForward = self.adModelForward
   adCriterionForward = self.adCriterionForward

   local dcostDx = autograd(adCost, {optimize = true})

   local dcostDxValue, _ = dcostDx(x, self.adParams, y)
   local xAdv = x + EPS * dcostDxValue/torch.norm(dcostDxValue)
   return xAdv
end
--------------------------------
--- END Adversarial examples ---
--------------------------------

