require 'Model'

local ModelBasicCNN, parent = torch.class('nn.ModelBasicCNN', 'nn.Model')


function ModelBasicCNN:__init(no_class_labels, use_existing, opt_run_on_cuda)
    parent.__init(self)

    self.model_path = "data/models/basic_cnn.t7"

    local opt = {}
    opt.no_class_labels = no_class_labels

    self:__load_model(use_existing, opt_run_on_cuda, opt)
end


function ModelBasicCNN:__createModel(opt)
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
    net:add(nn.Linear(200, opt.no_class_labels))
    net:add(nn.LogSoftMax())

    -- initialize
    local params, _ = net:getParameters()
    params:normal(0, 0.1)

    return net
end




