require 'nn'

local Model = torch.class('nn.Model')


function Model:__init()
end

function Model:__load_model(opt_run_on_cuda, opt)
    if not self.model_path then
        self.model_path = "data/models/generic_model.t7"
    end

    local net = self:__createModel(opt)

    -- init
    self.net = net
    self.criterion = nn.ClassNLLCriterion()
    self:run_on_cuda(opt_run_on_cuda)
    self.flatten_params, self.flatten_dloss_dparams = self.net:getParameters()
end


function Model:__createModel(opt)
    -- should be implemented in each model
    assert(false)
    return nil
end

function Model:save_me(obj_path)
    print("save obj to path", obj_path)
    local new_obj = {}
    new_obj.net = self.net
    new_obj.criterion = self.criterion

    torch.save(obj_path, new_obj)
end


function Model:load_me(obj_path)
    if paths.filep(obj_path) then
        print("load obj from path", obj_path)

        local load_obj = torch.load(obj_path)
        self.net = load_obj.net
        self.criterion = load_obj.criterion

        return true
    end
    return false
end


function Model:run_on_cuda(run)
    if run then
        self.net:cuda()
        self.criterion:cuda()
    else
        self.net:float()
        self.criterion:float()
    end
    self.flatten_params, self.flatten_dloss_dparams = self.net:getParameters()
end


-----------
-- Feval --
-----------
function Model:feval(inputs, labels)
    return function(x)
        self:__start_feval(x)
        local loss, dloss = self:__fwd_bckw_feval(inputs, labels)
        return loss, dloss
    end
end


function Model:__start_feval(x)
    if x ~= self.flatten_params then
        self.flatten_params:copy(x)
    end
    self.flatten_dloss_dparams:zero()
end



function Model:__fwd_bckw_feval(inputs, labels)
    -- compute the loss
    local outputs = self.net:forward(inputs)
    local loss = self.criterion:forward(outputs, labels)

    -- backpropagate the loss
    local dloss = self.criterion:backward(outputs, labels)
    local grad_input = self.net:backward(inputs, dloss)

    return loss, self.flatten_dloss_dparams
end


function Model:forward(inputs, labels)
    -- compute the loss
    local outputs = self.net:forward(inputs)
    local loss = self.criterion:forward(outputs, labels)

    return loss
end