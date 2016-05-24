--
-- A dataset
-- * has 3 splits: dataset.trainset, dataset.testset, dataset.validset
-- *
require 'paths'
require 'Dataset'


local DatasetCifarSmall, parent = torch.class('nn.DatasetCifarSmall', 'nn.Dataset')


function DatasetCifarSmall:__init(opt_run_on_cuda)
    parent.__init(self)
    self:__cifar10()
    self:run_on_cuda(opt_run_on_cuda)
end


function DatasetCifarSmall:__download_cifar10(dataset_path)
    local result = false
    local dataset_exists = paths.dirp(dataset_path)
    if not dataset_exists then
        os.execute('git clone https://github.com/soumith/cifar.torch ; cd cifar.torch ; th Cifar10BinToTensor.lua')

        os.execute('mkdir -p data/cifar10/')
        os.execute('mv cifar.torch/cifar10-train.t7 data/cifar10/')
        os.execute('mv cifar.torch/cifar10-test.t7 data/cifar10/')

        result = true

        dataset_exists = paths.dirp(dataset_path)
        if not dataset_exists then
            print('Error occured when downloading dataset. See download_cifar10 function')
            delete_cifar10(dataset_path)
        else
            delete_cifar10("cifar.torch")
        end
    end

    return result
end


function DatasetCifarSmall:__delete_cifar10(dataset_path)
    os.execute('rm -rf ' .. dataset_path)
end


function DatasetCifarSmall:__raw_cifar10_splits(trainset_path, testset_path)
    --[[
    Load torch datasets. Increment class labels (in torch e indexare de la 1).
    ]]--
    local train_and_valid_sets = torch.load(trainset_path)
    local testset = torch.load(testset_path)

    -- fix labels
    train_and_valid_sets.label = train_and_valid_sets.label + 1
    testset.label = testset.label + 1

    local trainset, validset = self:__random_split_train_valid(train_and_valid_sets)
    return {trainset, validset, testset}
end


function DatasetCifarSmall:__cifar10()
    print('Loading dataset...')

    local DATASET_NAME = 'cifar10'
    local DATASET_PATH = string.format('%s%s/', self.DATA_PATH, DATASET_NAME)
    local PREPROC_DATASET_PATH = DATASET_PATH .. 'cifar10-preproc.t7'

    local loaded = self:load_me(PREPROC_DATASET_PATH)
    if loaded then
        return
    end

    local trainset_path = string.format('%scifar10-train.t7', DATASET_PATH)
    local testset_path = string.format('%scifar10-test.t7', DATASET_PATH)
    local class_labels = {'airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

    local mean, stdv

    local just_downloaded = self:__download_cifar10(DATASET_PATH)
    local all_splits = self:__raw_cifar10_splits(trainset_path, testset_path)

    for split_idx = 1, 3 do
        local split = all_splits[split_idx]
        self:__comply_to_interface(split)
        self:__convert(split)
        mean, stdv = self:__normalize(split)
    end


    self.trainset = all_splits[1]
    self.validset = all_splits[2]
    self.testset = all_splits[3]
    self.class_labels = class_labels

    self:save_me(PREPROC_DATASET_PATH)
end

