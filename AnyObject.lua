require 'nn'
local AnyObject = torch.class('nn.AnyObject')


function AnyObject:save_me(obj_path)
    print("save obj to path", obj_path)
    local new_obj = {}
    for key, value in pairs(self) do
        new_obj[key] = value
    end
    torch.save(obj_path, new_obj)
end


function AnyObject:load_me(obj_path)
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