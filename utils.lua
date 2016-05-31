require 'cutorch'


function parseCmd(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options')
   cmd:option('-runOnCuda', true, 'Whether to run on GPU, using CudaTensors or on CPU, using FloatTensors. Default it runs on GPU.')
   cmd:option('-verbose', true, 'Print training intermediate results.')
   cmd:option('-adversarial', true, 'Evaluate both normal and adversarial loss.')
   cmd:text()

   -- parse input parameters
   local opt = cmd:parse(arg)
   return opt
end


function init(arg)
   local opt = parseCmd(arg)
   torch.setdefaulttensortype('torch.FloatTensor')
   cutorch.setDevice(4)

   return opt
end