'''
Created on Jun 17, 2024

@author: voodoocode
'''

import os
import mne
import mne.bem

import finnpy.src_rec.extract_anatomy
import threadpoolctl

FS_PATH = "/usr/local/freesurfer/7.4.1/"
VERBOSE_LVL = "ERROR"

PATIENT_NAME = "demo_patient"
ANATOMY_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/mne/anatomy/"
SENSOR_SPACE_REC_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/mne/raw_data/demo_patient/meg/rec.fif"
T1_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/finnpy/raw_data/demo_patient/mri/T1.mgz"

EMPTY_ROOM_PATH = "/home/voodoocode/Downloads/tmp11/empty_room/rec.fif"

TMP_DATA_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/mne/tmp/"
FS = 240

def pipeline(step):
    threadpoolctl.threadpool_limits(1, user_api='blas')
    
    os.environ["FREESURFER_HOME"]   = FS_PATH
    os.environ["FSFAST_HOME"]       = FS_PATH + "fsfast/"
    os.environ["FSF_OUTPUT_FORMAT"] = "nii.gz"
    os.environ["SUBJECTS_DIR"]      = ANATOMY_PATH
    os.environ["MNI_DIR"]           = FS_PATH + "mni/"
    
    os.environ["PATH"] = os.environ["PATH"]+":"+os.environ['FREESURFER_HOME']+"bin/"
    os.environ["PATH"] = os.environ["PATH"]+":"+os.environ['FSFAST_HOME']+"bin/"
    
    mne.set_config("FREESURFER_HOME", FS_PATH)
    
    if (step == "extract_anatomy"):
        finnpy.src_rec.extract_anatomy.extract_mri_anatomy(ANATOMY_PATH, PATIENT_NAME, T1_PATH, overwrite = False)
        mne.bem.make_watershed_bem(subject = PATIENT_NAME, subjects_dir = ANATOMY_PATH, overwrite = True, volume = "T1",
                                   atlas = False, gcaatlas = False, preflood = 25, show = False,
                                   copy = False, T1 = None, brainmask = "ws.mgz", verbose = VERBOSE_LVL)
    if (step == "calc_inv"):
        print("sen_cov")
        raw = mne.io.read_raw_fif(EMPTY_ROOM_PATH, preload = True, verbose = "ERROR")
        sensor_noise_cov = mne.compute_raw_covariance(raw, reject = dict(grad=4000e-13, mag=4e-12), method = "empirical",
                                                      verbose = "ERROR", n_jobs = 30)
        
        print("coreg")
        info = mne.io.read_info(SENSOR_SPACE_REC_PATH, verbose = VERBOSE_LVL)
        coreg = mne.coreg.Coregistration(info, PATIENT_NAME, ANATOMY_PATH, fiducials = "auto")
        coreg.fit_fiducials()
        coreg.fit_icp(n_iterations = 100, nasion_weight = 2., verbose = VERBOSE_LVL)
        coreg.omit_head_shape_points(distance = 5. / 1000)
        coreg.fit_icp(n_iterations = 100, nasion_weight = 10., verbose = VERBOSE_LVL)
        
        print("src_space")
        src_space = mne.setup_source_space(subject = PATIENT_NAME, spacing = "oct6", surface = "white", subjects_dir = ANATOMY_PATH, add_dist = True, n_jobs = 1, verbose = VERBOSE_LVL)
        
        print("bem")
        conductivity = (0.3,)#, 0.006, 0.3)
        bem_model = mne.make_bem_model(subject = PATIENT_NAME, ico = 4, conductivity = conductivity, subjects_dir = ANATOMY_PATH, verbose = VERBOSE_LVL)
        bem_solution = mne.make_bem_solution(bem_model, verbose = VERBOSE_LVL)

        #mne.bem.make_scalp_surfaces(subject = PATIENT_NAME, subjects_dir = ANATOMY_PATH, force = False, overwrite = False, no_decimate = True, verbose = VERBOSE_LVL)

        
        print("fwd")
        fwd = mne.make_forward_solution(info = info, trans = coreg.trans, src = src_space, bem = bem_solution, meg = True, eeg = False, mindist = 0.0, n_jobs = 1, verbose = VERBOSE_LVL)
        print("reset_fwd")
        fwd = mne.convert_forward_solution(fwd, surf_ori = True, force_fixed = True, copy = True, use_cps = True, verbose = VERBOSE_LVL)

        print("inv")
        inv = mne.minimum_norm.make_inverse_operator(info, fwd, sensor_noise_cov, loose = "auto", depth = None,
                                                     fixed = True, rank = "full", use_cps = True, verbose = VERBOSE_LVL)

def main():
    #pipeline("extract_anatomy")
    pipeline("calc_inv")

import time
import shutil

time.sleep(10)
for idx in range(4):
    shutil.move("/home/voodoocode/Downloads/tmp11/anatomy/demo_patient_mne_" + str(idx + 1),
                "/home/voodoocode/Downloads/tmp11/anatomy/demo_patient")
    start = time.time_ns()
    main()
    print(time.time_ns() - start)
    time.sleep(10)
    shutil.rmtree("/home/voodoocode/Downloads/tmp11/anatomy/demo_patient")





