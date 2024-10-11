'''
Created on Jun 17, 2024

@author: voodoocode
'''

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import threadpoolctl
threadpoolctl.threadpool_limits(1, user_api=None)

import torch
torch.set_num_threads(1)

import mne
import numpy as np

import finnpy.filters.frequency as ff
import finnpy.basic.downsampling as ds

import finnpy.file_io.data_manager as dm
import finnpy.src_rec.sen_cov
import finnpy.src_rec.coreg
import finnpy.src_rec.skull_skin_mdls
import finnpy.src_rec.bem_mdl
import finnpy.src_rec.cort_mdl
import finnpy.src_rec.fwd_mdl
import finnpy.src_rec.inv_mdl
import finnpy.src_rec.extract_anatomy
import finnpy.src_rec.subj_to_fsavg
import shutil

#################
# GENERAL PATHS #
#################
FREESURFER_PATH = "/usr/local/freesurfer/7.4.1/"
FASTSURFER_PATH = "/home/voodoocode/Downloads/tmp11/FastSurfer/"
FASTSURFER_PYTHON_PATH = "/home/voodoocode/local_python/bin/python"
FREESURFER_LICENSE_PATH = "/usr/local/freesurfer/7.4.1/license.txt"
ANATOMY_PATH = "/home/voodoocode/Downloads/tmp11/anatomy/"
EMPTY_ROOM_PATH = "/home/voodoocode/Downloads/tmp11/empty_room/rec.fif"

##########################
# SUBJECT SPECIFIC PATHS #
##########################
SUBJ_NAME = "demo_patient"
T1_PATH = "/home/voodoocode/Downloads/tmp11/anatomy/sub-Z446_T1w.nii.gz"
SENSOR_SPACE_REC_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/finnpy/raw_data/demo_patient/meg/rec.fif"
COV_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/finnpy/cov/"

####################
# FINNPY SPECIFICS #
####################
TMP_DATA_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/finnpy/tmp/"
FS = 240

import time

def pipeline(step):
    threadpoolctl.threadpool_limits(1, user_api='blas')
    
    if (step == "extract_anatomy"):
        overwrite_fs_extract = True
        overwrite_ws_extract = True
        finnpy.src_rec.extract_anatomy.init_paths(FREESURFER_PATH, FASTSURFER_PATH, FASTSURFER_PYTHON_PATH, FREESURFER_LICENSE_PATH, ANATOMY_PATH)
        finnpy.src_rec.extract_anatomy.extract_mri_anatomy(ANATOMY_PATH, SUBJ_NAME, T1_PATH, mode = "FreeSurfer", overwrite = overwrite_fs_extract)
        finnpy.src_rec.extract_anatomy.extract_skull_skin(ANATOMY_PATH, SUBJ_NAME, preflood_height = 25, overwrite = overwrite_ws_extract)
        finnpy.src_rec.extract_anatomy.calc_head_model(ANATOMY_PATH, SUBJ_NAME)
    if (step == "calc_inv"):
        print("sen_cov")
        overwrite_sen_cov = True
        sen_cov = finnpy.src_rec.sen_cov.run(EMPTY_ROOM_PATH, COV_PATH, method = None, float_sz = 64, method_params = None, overwrite = overwrite_sen_cov)
        
        rec_meta_info = mne.io.read_info(SENSOR_SPACE_REC_PATH)
        print("coreg"); (coreg, _) = finnpy.src_rec.coreg.run(SUBJ_NAME, ANATOMY_PATH, rec_meta_info)
        print("skull_skin_mdl"); skull_skin_mdl = finnpy.src_rec.skull_skin_mdls.read(ANATOMY_PATH, SUBJ_NAME, "MEG", coreg)
        print("bem_mdl"); bem_mdl = finnpy.src_rec.bem_mdl.run(FREESURFER_PATH, skull_skin_mdl.in_skull_vert, skull_skin_mdl.in_skull_faces)
        print("cort_mdl"); cort_mdl = finnpy.src_rec.cort_mdl.get(ANATOMY_PATH, SUBJ_NAME, coreg, bem_mdl.vert)
        print("fwd_mdl"); fwd_mdl = finnpy.src_rec.fwd_mdl.compute(cort_mdl, coreg, rec_meta_info, bem_mdl)
        print("rest_fwd_sol"); rest_fwd_sol = finnpy.src_rec.fwd_mdl.restrict(cort_mdl, fwd_mdl, coreg)
        print("inv_mdl"); inv_mdl = finnpy.src_rec.inv_mdl.compute(sen_cov, rest_fwd_sol, rec_meta_info)
        #print("subj_to_fsavg_mdl"); subj_to_fsavg_mdl = finnpy.src_rec.subj_to_fsavg.compute(cort_mdl, ANATOMY_PATH, SUBJ_NAME, FREESURFER_PATH, overwrite = True)

def main():
    #pipeline(step = "extract_anatomy")
    pipeline(step = "calc_inv")

time.sleep(10)
for idx in range(4):
    shutil.move("/home/voodoocode/Downloads/tmp11/anatomy/demo_patient_finnpy_" + str(idx + 1),
                "/home/voodoocode/Downloads/tmp11/anatomy/demo_patient")
    start = time.time_ns()
    main()
    print(time.time_ns() - start)
    time.sleep(10)
    shutil.rmtree("/home/voodoocode/Downloads/tmp11/anatomy/demo_patient")


