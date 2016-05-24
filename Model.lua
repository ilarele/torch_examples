require 'nn'
local Model = torch.class('nn.Model')


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