'''
Created on Jun 17, 2024

@author: voodoocode
'''

import threadpoolctl
import mne
import numpy as np
import os

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
import finnpy.src_rec.freesurfer
import finnpy.src_rec.subj_to_fsavg
import finnpy.src_rec.avg_src_reg

#################
# GENERAL PATHS #
#################
FS_PATH = "/usr/local/freesurfer/7.4.1/"
ANATOMY_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/finnpy/anatomy/"
EMPTY_ROOM_PATH = "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/data/finnpy/empty_room/rec.fif"

##########################
# SUBJECT SPECIFIC PATHS #
##########################
SUBJ_NAME = "demo_patient"
T1_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/finnpy/raw_data/demo_patient/mri/T1.mgz"
SENSOR_SPACE_REC_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/finnpy/raw_data/demo_patient/meg/rec.fif"
COV_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/finnpy/cov/"

####################
# FINNPY SPECIFICS #
####################
TMP_DATA_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/finnpy/tmp/"
FS = 240

import time

def pipeline(step):
    #Quite important, since blas does omes parallelization under the hood with interferes a lot with 1 external parallelization and 2 the test setup
    threadpoolctl.threadpool_limits(1, user_api='blas')
    #threadpoolctl.threadpool_limits(32, user_api='blas')
    
    if (step == "device"):
        ##############################
        # DEVICE/SETUP SPECIFIC PART #
        ##############################
        overwrite_sen_cov = True
        start = time.time_ns()
        finnpy.src_rec.sen_cov.run(EMPTY_ROOM_PATH, COV_PATH, method = None, float_sz = 64, method_params = None, overwrite = overwrite_sen_cov)
        print((time.time_ns() - start)/1e9)
        start = time.time_ns()
        finnpy.src_rec.sen_cov.run(EMPTY_ROOM_PATH, COV_PATH, method = None, float_sz = 128, method_params = None, overwrite = overwrite_sen_cov)
        print((time.time_ns() - start)/1e9)
        start = time.time_ns()
        finnpy.src_rec.sen_cov.run(EMPTY_ROOM_PATH, COV_PATH, method = None, float_sz = 256, method_params = None, overwrite = overwrite_sen_cov)
        print((time.time_ns() - start)/1e9)
        
    if (step == "subject"):
        #########################
        # SUBJECT SPECIFIC PART #
        #########################
        overwrite_fs_extract = False
        overwrite_ws_extract = False
        finnpy.src_rec.freesurfer.init_fs_paths(FS_PATH, ANATOMY_PATH)
        finnpy.src_rec.freesurfer.extract_mri_anatomy(ANATOMY_PATH, SUBJ_NAME, T1_PATH, overwrite = overwrite_fs_extract)
        finnpy.src_rec.freesurfer.extract_skull_skin(ANATOMY_PATH, SUBJ_NAME, preflood_height = 25, overwrite = overwrite_ws_extract)
        finnpy.src_rec.freesurfer.calc_head_model(ANATOMY_PATH, SUBJ_NAME)
        finnpy.src_rec.subj_to_fsavg.prepare(FS_PATH, ANATOMY_PATH, SUBJ_NAME)
    
    if (step == "session"):
        #########################
        # SESSION SPECIFIC PART #
        #########################
        sen_cov = finnpy.src_rec.sen_cov.load(COV_PATH)
        rec_meta_info = mne.io.read_info(SENSOR_SPACE_REC_PATH)
        
        if (os.path.exists(TMP_DATA_PATH + "coreg")):
            coreg = dm.load(TMP_DATA_PATH + "coreg")
        else:
            print("Computing coreg")
            (coreg, _) = finnpy.src_rec.coreg.run(SUBJ_NAME, ANATOMY_PATH, rec_meta_info)
            dm.save(coreg, TMP_DATA_PATH + "coreg")
            
        if (os.path.exists(TMP_DATA_PATH + "skull_skin_mdl")):
            skull_skin_mdl = dm.load(TMP_DATA_PATH + "skull_skin_mdl")
        else:
            print("Computing skin skull mdls")
            skull_skin_mdl = finnpy.src_rec.skull_skin_mdls.read(ANATOMY_PATH, SUBJ_NAME, "MEG", coreg)
            dm.save(skull_skin_mdl, TMP_DATA_PATH + "skull_skin_mdl")
            
        if (os.path.exists(TMP_DATA_PATH + "bem_mdl")):
            bem_mdl = dm.load(TMP_DATA_PATH + "bem_mdl")
        else:
            print("Computing BEM mdl")
            bem_mdl = finnpy.src_rec.bem_mdl.run(FS_PATH, skull_skin_mdl.in_skull_vert, skull_skin_mdl.in_skull_faces)
            dm.save(bem_mdl, TMP_DATA_PATH + "bem_mdl")
            
        if (os.path.exists(TMP_DATA_PATH + "cort_mdl")):
            cort_mdl = dm.load(TMP_DATA_PATH + "cort_mdl")
        else:
            print("Computing cort mdl")
            cort_mdl = finnpy.src_rec.cort_mdl.get(ANATOMY_PATH, SUBJ_NAME, coreg, bem_mdl.vert)
            dm.save(cort_mdl, TMP_DATA_PATH + "cort_mdl")
            
        if (os.path.exists(TMP_DATA_PATH + "fwd_mdl")):
            fwd_mdl = dm.load(TMP_DATA_PATH + "fwd_mdl")
        else:
            print("Computing fwd mdl")
            fwd_mdl = finnpy.src_rec.fwd_mdl.compute(cort_mdl, coreg, rec_meta_info, bem_mdl)
            dm.save(fwd_mdl, TMP_DATA_PATH + "fwd_mdl")
            
        if (os.path.exists(TMP_DATA_PATH + "rest_fwd_mdl")):
            rest_fwd_sol = dm.load(TMP_DATA_PATH + "rest_fwd_mdl")
        else:
            print("Computing rest fwd mdl")
            rest_fwd_sol = finnpy.src_rec.fwd_mdl.restrict(cort_mdl, fwd_mdl, coreg)
            dm.save(rest_fwd_sol, TMP_DATA_PATH + "rest_fwd_mdl")
            
        if (os.path.exists(TMP_DATA_PATH + "inv_mdl")):
            inv_mdl = dm.load(TMP_DATA_PATH + "inv_mdl")
        else:
            print("Computing inv mdl")
            inv_mdl = finnpy.src_rec.inv_mdl.compute(sen_cov, rest_fwd_sol, rec_meta_info)
            dm.save(inv_mdl, TMP_DATA_PATH + "inv_mdl")
            
        if (os.path.exists(TMP_DATA_PATH + "subj_to_fsavg")):
            subj_to_fsavg_mdl = dm.load(TMP_DATA_PATH + "subj_to_fsavg")
        else:
            print("Computing subj to fsavg")
            subj_to_fsavg_mdl = finnpy.src_rec.subj_to_fsavg.compute(cort_mdl, ANATOMY_PATH, SUBJ_NAME, FS_PATH, overwrite = False)
            dm.save(subj_to_fsavg_mdl, TMP_DATA_PATH + "subj_to_fsavg")
    
    if (step == "apply"):
        ###########################
        # APPLY INVERSE TRANSFORM #
        ###########################
        inv_mdl = dm.load(TMP_DATA_PATH + "inv_mdl")
        subj_to_fsavg_mdl = dm.load(TMP_DATA_PATH + "subj_to_fsavg")
        if (os.path.exists(TMP_DATA_PATH + "pp_data")):
            pp_data = dm.load(TMP_DATA_PATH + "pp_data")
        else:
            meta = mne.io.read_raw_fif(SENSOR_SPACE_REC_PATH, verbose='ERROR', on_split_missing = "ignore")
            data = meta.get_data()
            data = data[np.argwhere([("MEG" in name) for name in meta.info["ch_names"]]).squeeze(1), :]
            pp_data = np.zeros((data.shape[0], int(data.shape[1] / meta.info["sfreq"] * FS)))
            for channel_idx in np.arange(data.shape[0]):
                tmp_data = ff.fir(data[channel_idx, :], 1, FS//2, 1, meta.info["sfreq"])
                tmp_data = ds.run(tmp_data, meta.info["sfreq"], FS)
                
                pp_data[channel_idx, :] = tmp_data
            dm.save(pp_data, TMP_DATA_PATH + "pp_data")
        
        data_epochs = np.split(pp_data, np.arange(0, pp_data.shape[1], 240*3*60), axis = 1)
        for data_epoch in data_epochs:
            if (data_epoch.shape[1] == 0):
                continue
            src_data_epoch = finnpy.src_rec.inv_mdl.apply(data_epoch, inv_mdl)
            fs_avg_src_data_epoch = finnpy.src_rec.subj_to_fsavg.apply(subj_to_fsavg_mdl, src_data_epoch)
               
            #===================================================================
            # #Following line is funnpy exlusive, hence inactive, originally included to check results to previous stuff
            # (morphed_data_epoch, _, _) = finnpy.src_rec.avg_src_reg.run(fs_avg_src_data_epoch, subj_to_fsavg_mdl, FS_PATH)
            # ref_data = np.asarray(dm.load("/mnt/data/Professional/UHN/projects/data/MEG-AD2/recordings/src/al0095/a/tsss/01_OFF_tsss_1.fif")[0])
            # print((np.abs(morphed_data_epoch[:, 0:1000] - ref_data[:, 0:1000]) < 1e-4).all())
            # quit()
            #===================================================================

def visualization(mode):
    if (mode == "skin_skull"):
        ##########################
        # PLOT SKIN SKULL MODELS #
        ##########################
        models = finnpy.src_rec.skull_skin_mdls.read(ANATOMY_PATH, SUBJ_NAME, "full")
        finnpy.src_rec.skull_skin_mdls.plot(models, ANATOMY_PATH, SUBJ_NAME, block = True)
    
    
    if (mode == "coreg"):
        ##############
        # PLOT COREG #
        ##############
        rec_meta_info = mne.io.read_info(SENSOR_SPACE_REC_PATH)
        (coreg, bad_hsp_pts) = finnpy.src_rec.coreg.run(SUBJ_NAME, ANATOMY_PATH, rec_meta_info)
        meg_ref_pts = finnpy.src_rec.coreg.load_meg_ref_pts(rec_meta_info)
        finnpy.src_rec.coreg.plot_coregistration(coreg, meg_ref_pts, bad_hsp_pts, ANATOMY_PATH, SUBJ_NAME)

def main():
    pipeline(step = "device")
    #pipeline(step = "subject")
    #pipeline(step = "session")
    #pipeline(step = "apply")
    #visualization(mode = "skin_skull")
    #visualization(mode = "coreg")


main()
