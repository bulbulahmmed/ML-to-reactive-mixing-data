import numpy as np
import os

def write_inputfile(params,inputfilename):
    with open(inputfilename,'w') as f:
        for key in params.keys():
            params[key] = str(params[key])
            f.write('%s  %s\n' %(key,params[key]))

tau_list = [0.1, 0.2, 0.3, 0.4, 0.5]
alpha_T_list = [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.0]
k_f_list = [1, 2, 3, 4, 5]
v_0_list = alpha_T_list[:]
diff_list = [0.0,1.e-1,1.e-2,1.e-3]

counter = 1
for tau in tau_list:
    for alpha_T in alpha_T_list:
        for k_f in k_f_list:
            for v_0 in v_0_list:
                for diff in diff_list:
                    os.system('mkdir R' + str(counter))
                    inputfilename = 'R' + str(counter) + '/' + str(counter) + 'F.in'
                    counter = counter + 1
                    params = {}
                    params['ic_filename'] = 'T3_81_81_F.ic'
                    params['mesh_filename']  = 'T3_81_81.inp'
                    params['nonnegative_solver'] = ''
                    params['final_dt'] = 1.e-3
                    params['final_t'] = 1.0
                    params['v0'] = v_0
                    params['k_f'] = k_f
                    params['diffusivity'] = diff
                    params['tau'] = tau * 1.e-3
                    params['alpha_L'] = 1.0
                    params['alpha_T'] = alpha_T
                    write_inputfile(params, inputfilename)
