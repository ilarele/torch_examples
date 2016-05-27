require 'nn'

local Trainer = torch.class('nn.Trainer')


---------------
---- Train ----
---------------
function Trainer:train(trainset, validset, verbose)
    print('[Training...]')

    local ds_size = trainset:size(1)
    local no_iters = self:__get_no_iters(ds_size)
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

            -- update params with self.optim_function rules
            local _, fs, _ = self.optim_function(feval, model.flatten_params, self.optim_params)

            -- update loss
            epoch_loss = epoch_loss + fs[1]
        end

        -- report average error on epoch
        epoch_loss = epoch_loss / no_iters
        total_loss = total_loss + epoch_loss
        self:__logging(epoch .. " train_loss   " .. epoch_loss, verbose)
        self:test(validset, verbose)
    end

    local avg_loss = total_loss / self.no_epochs
    return avg_loss
end
---------------
-- End Train --
---------------


--------------
---- Test ----
--------------
function Trainer:test(testset, test_adversarial, verbose)
    -- print('[Testing...]')

    local ds_size = testset:size(1)
    local no_iters = self:__get_no_iters(ds_size)
    local perm_idx = torch.randperm(ds_size, 'torch.LongTensor')

    local avg_loss = 0
    local avg_loss_adv = 0
    for iter = 1, no_iters do
        -- get mini-batch
        local inputs, labels = self:__next_batch(testset, iter, perm_idx)
        local _, iter_loss = self.model:forward(inputs, labels)

        -- evaluate loss on this mini-batch
        if test_adversarial then
            local inputs_adv = self.model:adversarial_samples(inputs, labels)
            local _, iter_loss_adv = self.model:forward(inputs_adv, labels)
            avg_loss_adv = avg_loss_adv + iter_loss_adv
        end

        -- update loss
        avg_loss = avg_loss + iter_loss
    end

    -- avg loss
    avg_loss = avg_loss / no_iters
    avg_loss_adv = avg_loss_adv / no_iters

    self:__logging("test_loss     " .. avg_loss, verbose)
    self:__logging("test_loss_adv " .. avg_loss_adv, verbose)
    return avg_loss
end
-----------------
---- END Test ---
-----------------

-----------
-- Utils --
-----------
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


function Trainer:__get_no_iters(ds_size)
    local no_iters = torch.floor(ds_size/self.mini_bs)
    if no_iters > self.max_iters then
        no_iters = self.max_iters
    end
    return no_iters
end


function Trainer:__logging(to_print, verbose)
    if verbose then
        print(to_print)
    end
end
---------------
-- END Utils --
---------------

