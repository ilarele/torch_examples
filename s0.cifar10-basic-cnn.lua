require 'utils'

local DatasetClass, ModelClass, TrainerClass


function custom_setup(arg)
    local opt = init(arg)
    opt.run_on_cuda = true
    opt.reload_model = true
    opt.verbose = true

    -----------------------------------
    -- setup dataset, model and trainer
    -----------------------------------
    require 'TrainerAdam'
    require 'ModelBasicCNN'
    require 'DatasetCifarSmall'

    DatasetClass = nn.DatasetCifarSmall
    ModelClass = nn.ModelBasicCNN
    TrainerClass = nn.TrainerAdam
    -----------------------------------

    return opt
end


function main(arg)
    -- setup options
    local opt = custom_setup(arg)

    -- initialize
    local dataset = DatasetClass(opt.run_on_cuda)
    local model = ModelClass(#dataset.class_labels, opt.reload_model, opt.run_on_cuda)
    local trainer = TrainerClass(model)

    -- train
    local train_loss = trainer:train(dataset.trainset, opt.verbose)
    local test_loss = trainer:test(dataset.testset, opt.verbose)
end


main(arg)
