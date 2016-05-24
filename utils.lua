function parse_cmd(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Options')
    cmd:option('-run_on_cuda', false, 'Whether to run on GPU, using CudaTensors. Default it runs on CPU, using FloatTensors.')
    cmd:text()

    -- parse input params
    local opt = cmd:parse(arg)
    return opt
end


function init(arg)
    local opt = parse_cmd(arg)
    torch.setdefaulttensortype('torch.FloatTensor')

    return opt
end