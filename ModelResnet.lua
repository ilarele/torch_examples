--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

require 'Model'

local ModelResnet, parent = torch.class('nn.ModelResnet', 'nn.Model')

local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization
local ReLU, Avg, Convolution

function ModelResnet:__init(noClassLabels, optRunOnCuda)
   self.model_path = "data/models/resnet.t7"

   local opt = {}
   opt.depth = 20
   opt.shortcutType = 'C'
   opt.noClassLabels = noClassLabels

   if optRunOnCuda then
      ReLU = cudnn.ReLU
      Avg = cudnn.SpatialAveragePooling
      Convolution = cudnn.SpatialConvolution
   else
      ReLU = nn.ReLU
      Avg = nn.SpatialAveragePooling
      Convolution = nn.SpatialConvolution
   end

   parent:__init(self, optRunOnCuda, opt)
end


function ModelResnet:__createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
        (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
        -- 1x1 convolution
        x_shortcut = nn.Sequential()
        x_shortcut:add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
        x_shortcut:add(SBatchNorm(nOutputPlane))
        return x_shortcut
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         x_shortcut = nn.Sequential()
         x_shortcut:add(nn.SpatialAveragePooling(1, 1, stride, stride))
         x_shortcut:add(nn.Concat(2):add(nn.Identity()):add(nn.MulConstant(0)))
         return x_shortcut
      else
         return nn.Identity()
      end
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride)
      local nInputPlane = iChannels
      iChannels = n

      local fx = nn.Sequential()
      fx:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      fx:add(SBatchNorm(n))
      fx:add(ReLU(true))
      fx:add(Convolution(n,n,3,3,1,1,1,1))
      fx:add(SBatchNorm(n))

      local sum_x_fx = nn.Sequential()
      sum_x_fx:add(nn.ConcatTable():add(fx):add(shortcut(nInputPlane, n, stride)))
      sum_x_fx:add(nn.CAddTable(true))
      sum_x_fx:add(ReLU(true))

      return sum_x_fx
   end

   -- The bottleneck residual layer for 50, 101, and 152 layer networks
   local function bottleneck(n, stride)
      local nInputPlane = iChannels
      iChannels = n * 4

      local fx = nn.Sequential()
      fx:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
      fx:add(SBatchNorm(n))
      fx:add(ReLU(true))
      fx:add(Convolution(n,n,3,3,stride,stride,1,1))
      fx:add(SBatchNorm(n))
      fx:add(ReLU(true))
      fx:add(Convolution(n, n*4, 1,1,1,1,0,0))
      fx:add(SBatchNorm(n * 4))

      local sum_x_fx = nn.Sequential()
      sum_x_fx:add(nn.ConcatTable():add(fx):add(shortcut(nInputPlane, n * 4, stride)))
      sum_x_fx:add(nn.CAddTable(true))
      sum_x_fx:add(ReLU(true))
      return sum_x_fx
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count, stride)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1))
      end
      return s
   end

   local model = nn.Sequential()
   -- Model type specifies number of layers for CIFAR-10 model
   assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
   local n = (depth - 2) / 6
   iChannels = 16
   print(' | ResNet-' .. depth .. ' CIFAR-10')

   -- The ResNet CIFAR-10 model
   model:add(Convolution(3,16,3,3,1,1,1,1))
   model:add(SBatchNorm(16))
   model:add(ReLU(true))
   model:add(layer(basicblock, 16, n))
   model:add(layer(basicblock, 32, n, 2))
   model:add(layer(basicblock, 64, n, 2))
   model:add(Avg(8, 8, 1, 1))
   model:add(nn.View(64):setNumInputDims(3))
   model:add(nn.Linear(64, 10))
   model:add(nn.LogSoftMax())

   -- init weights
   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         v.bias = nil
         v.gradBias = nil
      end
   end

   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias = nil
         v.gradBias = nil
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   -- uncomment this if you don't need grad_input
   -- tou will need it if you want to build adversarial examples
   -- model:get(1).gradInput = nil

   return model, nn.ClassNLLCriterion()
end

