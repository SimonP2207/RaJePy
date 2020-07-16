# -*- coding: utf-8 -*-
"""
RAdio JEts in PYthon test program.
"""
import os, sys, shutil
from RaJePy import JetModel, ModelRun

if __name__ == '__main__':
    jet_param_file = sys.argv[1]
    pline_param_file = sys.argv[2]
    
    # Testing files if needed
    # jet_param_file = '/Users/simon/Dropbox/Paper_RadioRT/Scripts/params/model-params-s255IRS3-l_z1.py'
    # pline_param_file = '/Users/simon/Dropbox/Paper_RadioRT/Scripts/params/pipeline-params-1time-100freqs.py'

    # Need function to determine whether to load from saved state or not
    pline = ModelRun(JetModel(jet_param_file), pline_param_file)

    if os.path.exists(pline.save_file):
        ans = ''
        message = "Saved pipeline detected. Resume? (y/n): "
        while True:
            ans = input(message)
            if ans in ('y', 'n'):
                break
            else:
                message = "Please enter either 'y' or 'n': "
                
        if ans == 'y':
            pline = ModelRun.load_pipeline(pline.save_file)
        pline.execute()
    else:
        pline.execute()
        
    for f in (jet_param_file, pline_param_file):
        dest = os.sep.join([pline.params['dcys']['model_dcy'],
                            os.path.basename(f)])
        shutil.copyfile(jet_param_file, dest)
