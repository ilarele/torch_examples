local nn = require 'nn'
local autograd = require 'autograd'


local Model = torch.class('nn.Model')


----------------
-- Init Model --
----------------
function Model:__init(opt_run_on_cuda, opt)
    local net, criterion = self:__createModel(opt)

    -- init
    self.net = net
    self.criterion = criterion
    self:run_on_cuda(opt_run_on_cuda)
    self.flatten_params, self.flatten_dloss_dparams = self.net:getParameters()

    -- autograd transofrmations, used for adversarial examples
    self.autograd_model_forward, self.autograd_params = autograd.functionalize(self.net)
    self.autograd_criterion_forward = autograd.functionalize(self.criterion)
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
        if x ~= self.flatten_params then
            self.flatten_params:copy(x)
        end
        self.flatten_dloss_dparams:zero()

        local outputs, loss = self:forward(inputs, labels)
        local dloss_dparams, grad_input = self:backward(inputs, outputs, labels)

        return loss, dloss_dparams, grad_input
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
    local grad_input = self.net:backward(inputs, dloss)

    return dloss, grad_input
end
---------------------------------
----------- END Feval -----------
---------------------------------


--------------------------------
------ Adversarial examples ----
--------------------------------
function autograd_cost(x, self, y)
    local y_pred = self.autograd_model_forward(self.autograd_params, x)
    local loss = self.autograd_criterion_forward(y_pred, y)
    return loss
end


local EPS = 100
function Model:adversarial_samples(x, y)
    local dcost_dx = autograd(autograd_cost, {optimize = true})

    local dcost_dx_value, _ = dcost_dx(x, self, y)
    local x_adv = x + EPS * dcost_dx_value/torch.norm(dcost_dx_value)
    return x_adv
end
--------------------------------
--- End Adversarial examples ---
--------------------------------

