require 'nn'

local Trainer = torch.class('nn.Trainer')

function Trainer:__init(model)
end

function Trainer:__next_batch(dataset, iter, perm_idx)
    --[[
    This should return a batch from the shuffled dataset: batch(dataset[perm_idx], start_index)
    - perm_idx: indexes permutation
    - start_index: it refers to the perm_idx
    ]]--
    local mini_bs = self.mini_bs
    local ds_size = dataset:size(1)
    local start_index = self.mini_bs * (iter - 1) + 1

    local crt_mini_bs = math.min(mini_bs, ds_size - start_index + 1)
    local end_index = crt_mini_bs + start_index - 1

    local shuffled_idx = perm_idx[{{start_index, end_index}}]
    local batch_idx = perm_idx:index(1, shuffled_idx)

    local inputs = dataset.data:index(1, batch_idx)
    local labels = dataset.label:index(1, batch_idx)

    return inputs, labels
end


function Trainer:get_no_iters(ds_size)
    local no_iters = torch.floor(ds_size/self.mini_bs)
    if no_iters > self.max_iters then
        no_iters = self.max_iters
    end
    return no_iters
end




function Trainer:train(trainset, verbose)
    print('[Training...]')

    local ds_size = trainset:size(1)
    local no_iters = self:get_no_iters(ds_size)
    local perm_idx = torch.randperm(ds_size, 'torch.LongTensor')
    local model = self.model
    local total_loss = 0

    for epoch = 1, self.no_epochs do
        local epoch_loss = 0
        for iter = 1, no_iters do

            -- get mini-batch
            local inputs, labels = self:__next_batch(trainset, iter, perm_idx)

            -- get feval for this batch and model
            local feval = model:feval(inputs, labels)

            -- for debugging inside feval, just call it (outside the optim function)
            -- feval(model.flatten_params)
            -- update params with self.optim_function rules
            local _, fs = self.optim_function(feval, model.flatten_params, self.optim_params)

            -- update loss
            epoch_loss = epoch_loss + fs[1]
        end

        -- report average error on epoch
        epoch_loss = epoch_loss / no_iters
        total_loss = total_loss + epoch_loss
        self:__logging(epoch .. " epoch_loss: " .. epoch_loss, verbose)
    end

    local avg_loss = total_loss / self.no_epochs
    return avg_loss
end



function Trainer:test(testset, verbose)
    print('[Testing...]')

    local ds_size = testset:size(1)
    local no_iters = self:get_no_iters(ds_size)
    local perm_idx = torch.randperm(ds_size, 'torch.LongTensor')

    local avg_loss = 0
    for iter = 1, no_iters do
        -- get mini-batch
        local inputs, labels = self:__next_batch(testset, iter, perm_idx)

        -- evaluate loss on this mini-batch
        local iter_loss = self.model:forward(inputs, labels)

        -- update loss
        avg_loss = avg_loss + iter_loss
    end

    -- avg loss
    avg_loss = avg_loss / no_iters
    self:__logging("test_loss " .. avg_loss, verbose)
    return avg_loss
end


function Trainer:__logging(to_print, verbose)
    if verbose then
        print(to_print)
    end
end

