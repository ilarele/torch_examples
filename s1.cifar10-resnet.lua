require 'utils'

local DatasetClass, ModelClass, TrainerClass


function custom_setup(arg)
    local cmd_opt = init(arg)
    cmd_opt.run_on_cuda = true
    cmd_opt.reload_model = true
    cmd_opt.verbose = true

    -----------------------------------
    -- setup dataset, model and trainer
    -----------------------------------
    require 'TrainerAdam'
    require 'ModelResnetSmallCifar'
    require 'DatasetCifarSmall'

    DatasetClass = nn.DatasetCifarSmall
    ModelClass = nn.ModelResnetSmallCifar
    TrainerClass = nn.TrainerAdam
    -----------------------------------

    return cmd_opt
end


function main(arg)
    local cmd_opt = custom_setup(arg)

    -- initialize
    local dataset = DatasetClass(cmd_opt.run_on_cuda)
    local model = ModelClass(#dataset.class_labels, cmd_opt.run_on_cuda)
    local trainer = TrainerClass(model)

    -- train
    local train_loss = trainer:train(dataset.trainset, dataset.validset, cmd_opt.verbose)
    local test_loss = trainer:test(dataset.testset, cmd_opt.verbose)
end


main(arg)
