require 'torch'

function parseCmd(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options')
   cmd:option('-runOnCuda', false, 'CudaTensors for CPU or FloatTensors for GPU. Default GPU.')
   cmd:option('-verbose', true, 'Print training intermediate results.')
   cmd:option('-printAdversarial', false, 'Evaluate both normal and adversarial loss.')
   cmd:option('-showImages', false, 'Show adversarial images. One at the end of each epoch.')
   cmd:text()

   -- parse input parameters
   local opt = cmd:parse(arg)
   return opt
end


function init(arg)
   local opt = parseCmd(arg)
   torch.setdefaulttensortype('torch.FloatTensor')

   if opt.runOnCuda then
      require 'cutorch'
      print "Runs on GPU"
   else
      print "Runs on CPU. For running on GPU, add -runOnCuda cmd line param"
   end

   if opt.showImages then
      opt.printAdversarial = true
   end

   if opt.printAdversarial then
      print "Run this using qlua (for showing normal images vs adversarial images)"
   end

   return opt
end