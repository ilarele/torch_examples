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

   -- autograd transofrmations, used for adversarial examples
   self.autogradModelForward, self.autogradParams = autograd.functionalize(self.net)
   self.autogradCriterionForward = autograd.functionalize(self.criterion)
end


function Model:runOnCuda(run)
   if run then
      self.net:cuda()
      self.criterion:cuda()
   else
      self.net:float()
      self.criterion:float()
   end
   self.flattenParams, self.flattenDlossDparams = self.net:getParameters()
end


function Model:__createModel(opt)
   -- should be implemented in each model
   assert(false)
   return nil
end

--------------
-- END Init --
--------------


--------------------
-- serialization ---
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
--- end serialization --
------------------------


---------------------------------
------------- Feval -------------
---------------------------------
function Model:feval(inputs, labels)
   -- forward the input
   -- backpropagate the loss
   -- return loss, params, dJ/dX
   return function(x)
      if x ~= self.flattenParams then
         self.flattenParams:copy(x)
      end
      self.flattenDlossDparams:zero()

      local outputs, loss = self:forward(inputs, labels)
      local dlossDparams, gradInput = self:backward(inputs, outputs, labels)

      return loss, dlossDparams, gradInput
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
   local dloss = self.criterion:backward(outputs, labels)
   local gradInput = self.net:backward(inputs, dloss)

   return dloss, gradInput
end
---------------------------------
----------- END Feval -----------
---------------------------------


--------------------------------
------ Adversarial examples ----
--------------------------------
function autogradCost(x, self, y)
   local y_pred = self.autogradModelForward(self.autogradParams, x)
   local loss = self.autogradCriterionForward(y_pred, y)
   return loss
end


local EPS = 100
function Model:adversarialSamples(x, y)
   local dcostDx = autograd(autogradCost, {optimize = true})

   local dcostDxValue, _ = dcostDx(x, self, y)
   local xAdv = x + EPS * dcostDxValue/torch.norm(dcostDxValue)
   return xAdv
end
--------------------------------
--- End Adversarial examples ---
--------------------------------

