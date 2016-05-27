--
-- A dataset
-- * has 3 splits: dataset.trainset, dataset.testset, dataset.validset

require 'paths'


local Dataset = torch.class('nn.Dataset')
local DATA_PATH = 'data/'


function Dataset:__init(opt_run_on_cuda)
    self.DATA_PATH = 'data/'
    self.TRAIN_VALIDATION_RATIO = 0.8
end


function Dataset:run_on_cuda(run)
    local splits = {"trainset", "validset", "testset"}

    if run then
        require 'cunn'
        require 'cudnn'

        for i = 1, 3 do
            self[splits[i]].data = self[splits[i]].data:cuda()
            self[splits[i]].label = self[splits[i]].label:cuda()
        end
    else
        -- float precision is good enough for ML
        for i = 1, 3 do
            self[splits[i]].data = self[splits[i]].data:float()
            self[splits[i]].label = self[splits[i]].label:float()
        end
    end
end


-----------------------------
-- torch dataset interface --
-----------------------------
function Dataset:__comply_to_interface(dataset_split)
    -- make it indexable []
    setmetatable(dataset_split,
        {__index = function(t, i)
                        return {t.data[i], t.label[i]}
                    end}
    );

    -- add size()
    function dataset_split:size()
        return self.data:size(1)
    end
end
-------------------------------------
---- END torch dataset interface ----
-------------------------------------



--------------------
---- Preprocess ----
--------------------
function Dataset:__random_split_train_valid(train_and_valid_sets)
    -- split train in train + validation
    torch.manualSeed(1)
    local ds_size = train_and_valid_sets.data:size(1)
    local perm_idx = torch.randperm(ds_size, 'torch.LongTensor')
    local train_idx = perm_idx[{{1, ds_size * self.TRAIN_VALIDATION_RATIO}}]
    local valid_idx = perm_idx[{{ds_size * self.TRAIN_VALIDATION_RATIO + 1, ds_size}}]

    -- trainset = train_and_valid_sets
    local trainset = {}
    trainset.data = train_and_valid_sets.data:index(1, train_idx)
    trainset.label = train_and_valid_sets.label:index(1, train_idx)

    local validset = {}
    validset.data = train_and_valid_sets.data:index(1, valid_idx)
    validset.label = train_and_valid_sets.label:index(1, valid_idx)

    return trainset, validset
end


function Dataset:__convert(dataset_split)
    dataset_split.data = dataset_split.data:float()
    dataset_split.label = dataset_split.label:long()
    return dataset_split
end


function Dataset:__normalize(split, mean, stdv)
    local mean = mean or {}
    local stdv = stdv or {}

    -- over each image channel (np.vectorize this)
    for i = 1, 3 do
        -- TODO: this is ugly, compute inplace
        if #mean == i - 1 or #stdv == i - 1 then
            -- mean and stdv estimation
            mean[i] = split.data[{ {}, {i}, {}, {}  }]:mean()
            stdv[i] = split.data[{ {}, {i}, {}, {}  }]:std()
        end

        -- mean subtraction and std scaling
        split.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
        split.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
    end
    return mean, stdv
end
------------------------
---- END Preprocess ----
------------------------


--------------------
-- serialization ---
--------------------
function Dataset:save_me(obj_path)
    print("save obj to path", obj_path)
    local new_obj = {}
    for key, value in pairs(self) do
        new_obj[key] = value
    end
    torch.save(obj_path, new_obj)
end


function Dataset:load_me(obj_path)
    if paths.filep(obj_path) then
        print("load obj from path", obj_path)
        local load_obj = torch.load(obj_path)
        for key, value in pairs(load_obj) do
            self[key] = value
        end
        return true
    end
    return false
end
------------------------
--- END serialization --
------------------------
