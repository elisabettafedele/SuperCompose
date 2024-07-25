from scipy.spatial.transform import Rotation as R
import json
import os

def save_sq_json(scale, rotate, trans, exps, save_path):
    num_el = scale.shape[0]
    components = []
    
    for k in range(num_el):
        component = {}
        component['scale'] = scale[k,:].tolist()
        rot3x3 = rotate[k,:]
        r = R.from_matrix(rot3x3)
        
        # convert from 3 x3 into euler angles
        component['rotation'] = rot3x3.tolist()
        component['position'] = trans[k,:].tolist()
        component['epsilon1'] = exps[k,0].tolist()
        component['epsilon2'] = exps[k,1].tolist()
        components.append(component)
        
    res = {}
    res['components'] = components
    json.dump(res, open(os.path.join(save_path,  'my_sq.json'), 'w'))