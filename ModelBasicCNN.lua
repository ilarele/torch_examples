require 'Model'

local ModelBasicCNN, parent = torch.class('nn.ModelBasicCNN', 'nn.Model')


function ModelBasicCNN:__init(no_class_labels, use_existing, opt_run_on_cuda)
    local MODEL_PATH = "data/models/basic_cnn.t7"
    if paths.filep(MODEL_PATH) and use_existing then
        self:load_me(MODEL_PATH)
        self:run_on_cuda(opt_run_on_cuda)
        self.flatten_params, self.flatten_dloss_dparams = self.net:getParameters()
    else
        ------------------------------------------------------------------------------
        -- net
        ------------------------------------------------------------------------------
        local net = nn.Sequential()
        -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
        net:add(nn.SpatialConvolution(3, 32, 5, 5))
        net:add(nn.ReLU())
        net:add(nn.SpatialMaxPooling(3, 3, 3, 3))
        -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
        net:add(nn.SpatialConvolution(32, 64, 5, 5))
        net:add(nn.ReLU())
        net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        -- stage 3 : standard 2-layer MLP:
        net:add(nn.Reshape(64*2*2))
        net:add(nn.Linear(64*2*2, 200))
        net:add(nn.ReLU())
        net:add(nn.Linear(200, no_class_labels))
        net:add(nn.LogSoftMax())

        ------------------------------------------------------------------------------
        -- INIT
        ------------------------------------------------------------------------------
        self.net = net
        self.criterion = nn.ClassNLLCriterion()
        self:run_on_cuda(opt_run_on_cuda)

        self.flatten_params, self.flatten_dloss_dparams = net:getParameters()
        self.flatten_params.normal(0, 0.1)
    end
end


function ModelBasicCNN:feval(inputs, labels)
    return function(x)
        self:__start_feval(x)
        local loss, dloss = self:__fwd_bw_feval(inputs, labels)
        return loss, dloss
    end
end


function ModelBasicCNN:__start_feval(x)
    if x ~= self.flatten_params then
        self.flatten_params:copy(x)
    end
    self.flatten_dloss_dparams:zero()
end


function ModelBasicCNN:__fwd_bw_feval(inputs, labels)
    -- compute the loss
    local outputs = self.net:forward(inputs)
    local loss = self.criterion:forward(outputs, labels)

    -- backpropagate the loss
    local dloss = self.criterion:backward(outputs, labels)
    local grad_input = self.net:backward(inputs, dloss)

    return loss, self.flatten_dloss_dparams
end


function ModelBasicCNN:forward(inputs, labels)
    -- compute the loss
    local outputs = self.net:forward(inputs)
    local loss = self.criterion:forward(outputs, labels)

    return loss
end


function ModelBasicCNN:run_on_cuda(run)
    if run then
        self.net:cuda()
        self.criterion:cuda()
        -- self.flatten_params = self.flatten_params:cuda()
        -- self.flatten_dloss_dparams = self.flatten_dloss_dparams:cuda()
        self.flatten_params, self.flatten_dloss_dparams = self.net:getParameters()
        -- self.flatten_params.normal(0, 0.1)
    else
        self.net:float()
        self.criterion:float()
        self.flatten_params = self.flatten_params:float()
        self.flatten_dloss_dparams = self.flatten_dloss_dparams:float()
    end
end

