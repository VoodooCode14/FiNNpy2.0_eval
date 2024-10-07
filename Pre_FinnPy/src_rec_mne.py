'''
Created on Jun 17, 2024

@author: voodoocode
'''

import os
import mne
import mne.bem
import numpy as np
import matplotlib.pyplot as plt

import finnpy.file_io.data_manager as dm
import finnpy.filters.frequency as ff
import finnpy.basic.downsampling as ds

import finnpy.src_rec.freesurfer
import threadpoolctl

FS_PATH = "/usr/local/freesurfer/7.4.1/"
VERBOSE_LVL = "ERROR"

PATIENT_NAME = "demo_patient"
ANATOMY_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/mne/anatomy/"
SENSOR_SPACE_REC_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/mne/raw_data/demo_patient/meg/rec.fif"
T1_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/finnpy/raw_data/demo_patient/mri/T1.mgz"

TMP_DATA_PATH = "/mnt/data/Professional/UHN/projects/data/finnpy2/mne/tmp/"
FS = 240

def pipeline(step):
    #Quite important, since blas does omes parallelization under the hood with interferes a lot with 1 external parallelization and 2 the test setup
    threadpoolctl.threadpool_limits(1, user_api='blas')
    
    os.environ["FREESURFER_HOME"]   = FS_PATH
    os.environ["FSFAST_HOME"]       = FS_PATH + "fsfast/"
    os.environ["FSF_OUTPUT_FORMAT"] = "nii.gz"
    os.environ["SUBJECTS_DIR"]      = ANATOMY_PATH
    os.environ["MNI_DIR"]           = FS_PATH + "mni/"
    
    os.environ["PATH"] = os.environ["PATH"]+":"+os.environ['FREESURFER_HOME']+"bin/"
    os.environ["PATH"] = os.environ["PATH"]+":"+os.environ['FSFAST_HOME']+"bin/"

    if ("FREESURFER_HOME" not in mne.get_config().keys()):
        mne.set_config("FREESURFER_HOME", FS_PATH)
        
    if (step == "device"):
        if (os.path.exists(TMP_DATA_PATH + "sensor_noise_cov")):
            sensor_noise_cov = dm.load(TMP_DATA_PATH + "sensor_noise_cov")
        else:
            file = "/mnt/data/Professional/UHN/projects/data/MEG_TRD/empty_room_filt/rec2/raw/tsss/2_tsss.fif"
            raw = mne.io.read_raw_fif(file, preload = True, verbose = "ERROR")
            sensor_noise_cov = mne.compute_raw_covariance(raw, reject = dict(grad=4000e-13, mag=4e-12), method = "empirical",
                                                          verbose = "ERROR", n_jobs = 30)
            dm.save(sensor_noise_cov, TMP_DATA_PATH + "sensor_noise_cov")
    
    if (step == "missing"):
        finnpy.src_rec.freesurfer.extract_mri_anatomy(ANATOMY_PATH, PATIENT_NAME, T1_PATH, overwrite = False)
    
    if (step == "fs"):
        if (os.path.exists(TMP_DATA_PATH + "src_space")):
            src_space = dm.load(TMP_DATA_PATH + "src_space")
        else:
            src_space = mne.setup_source_space(subject = PATIENT_NAME, spacing = "oct6", surface = "white", subjects_dir = ANATOMY_PATH, add_dist = True, n_jobs = 1, verbose = VERBOSE_LVL)
            dm.save(src_space, TMP_DATA_PATH + "src_space")
    
    if (step == "model"):
        src_space = dm.load(TMP_DATA_PATH + "src_space")
        sensor_noise_cov = dm.load(TMP_DATA_PATH + "sensor_noise_cov")
        if (os.path.exists(TMP_DATA_PATH + "bem_solution")):
            bem_solution = dm.load(TMP_DATA_PATH + "bem_solution")
        else:
            mne.bem.make_watershed_bem(subject = PATIENT_NAME, subjects_dir = ANATOMY_PATH, overwrite = True, volume = "T1",
                                       atlas = False, gcaatlas = False, preflood = 25, show = False,
                                       copy = False, T1 = None, brainmask = "ws.mgz", verbose = VERBOSE_LVL)
            conductivity = (0.3,)#, 0.006, 0.3)
            bem_model = mne.make_bem_model(subject = PATIENT_NAME, ico = 4, conductivity = conductivity, subjects_dir = ANATOMY_PATH, verbose = VERBOSE_LVL)
            bem_solution = mne.make_bem_solution(bem_model, verbose = VERBOSE_LVL)
            dm.save(bem_solution, TMP_DATA_PATH + "bem_solution")
        
        info = mne.io.read_info(SENSOR_SPACE_REC_PATH, verbose = VERBOSE_LVL)
        if (os.path.exists(TMP_DATA_PATH + "coreg")):
            coreg = dm.load(TMP_DATA_PATH + "coreg")
        else:
            mne.bem.make_scalp_surfaces(subject = PATIENT_NAME, subjects_dir = ANATOMY_PATH, force = False, overwrite = False, no_decimate = True, verbose = VERBOSE_LVL)
            
            coreg = mne.coreg.Coregistration(info, PATIENT_NAME, ANATOMY_PATH, fiducials = "auto")
            coreg.fit_fiducials()
            coreg.fit_icp(n_iterations = 100, nasion_weight = 2., verbose = VERBOSE_LVL)
            print(np.mean(coreg.compute_dig_mri_distances()))
            coreg.omit_head_shape_points(distance = 5. / 1000)
            coreg.fit_icp(n_iterations = 100, nasion_weight = 10., verbose = VERBOSE_LVL)
            print(np.mean(coreg.compute_dig_mri_distances()))
            dm.save(coreg, TMP_DATA_PATH + "coreg")
        
        fwd = mne.make_forward_solution(info = info, trans = coreg.trans, src = src_space, bem = bem_solution, meg = True, eeg = False, mindist = 0.0, n_jobs = 1, verbose = VERBOSE_LVL)
        fwd = mne.convert_forward_solution(fwd, surf_ori = True, force_fixed = True, copy = True, use_cps = True, verbose = VERBOSE_LVL)
        if (os.path.exists(TMP_DATA_PATH + "inv_mdl") == False):
            inv = mne.minimum_norm.make_inverse_operator(info, fwd, sensor_noise_cov, loose = "auto", depth = None,
                                                         fixed = True, rank = "full", use_cps = True, verbose = VERBOSE_LVL)
            dm.save(inv, TMP_DATA_PATH + "inv_mdl")
        else:
            inv = dm.load(TMP_DATA_PATH + "inv_mdl")
        if (os.path.exists(ANATOMY_PATH + "fsaverage/src_model-src.fif") == False):
            src_fsaverage = mne.setup_source_space(subject = "fsaverage", spacing = "oct6", surface = "white", subjects_dir = ANATOMY_PATH, verbose = VERBOSE_LVL)
            mne.write_source_spaces(ANATOMY_PATH + "fsaverage/src_model-src.fif", src_fsaverage, verbose = VERBOSE_LVL)
        else:
            src_fsaverage = mne.read_source_spaces(ANATOMY_PATH + "fsaverage/src_model-src.fif", verbose = VERBOSE_LVL)
    
    if (step == "apply"):
        inv = dm.load(TMP_DATA_PATH + "inv_mdl")
        src_fsaverage = mne.read_source_spaces(ANATOMY_PATH + "fsaverage/src_model-src.fif", verbose = VERBOSE_LVL)
        if (os.path.exists(TMP_DATA_PATH + "pp_data")):
            pp_data = dm.load(TMP_DATA_PATH + "pp_data")
            meta = mne.io.read_raw_fif(SENSOR_SPACE_REC_PATH, verbose='ERROR', on_split_missing = "ignore")
        else:
            meta = mne.io.read_raw_fif(SENSOR_SPACE_REC_PATH, verbose='ERROR', on_split_missing = "ignore")
            data = meta.get_data()
            pp_data = np.zeros((data.shape[0], int(data.shape[1] / meta.info["sfreq"] * FS)))
            for channel_idx in np.arange(data.shape[0]):
                tmp_data = ff.fir(data[channel_idx, :], 1, FS//2, 1, meta.info["sfreq"])
                tmp_data = ds.run(tmp_data, meta.info["sfreq"], FS)
                
                pp_data[channel_idx, :] = tmp_data
            dm.save(pp_data, TMP_DATA_PATH + "pp_data")
        
        meg_ind = np.argwhere([1 if ("MEG" in name) else 0 for name in meta.info["ch_names"]]).squeeze(1)
        ch_names = (np.asarray(meta.info["ch_names"])[meg_ind]).tolist()
        ch_types = (np.asarray(["mag" if (int(meta_ch["coil_type"]) == 3024) else "grad" for meta_ch in meta.info["chs"]])[meg_ind]).tolist()
        pp_data = pp_data[meg_ind, :]
        loc_info = mne.create_info(ch_names = ch_names, sfreq = FS, ch_types = ch_types)
        loc_source_data = mne.minimum_norm.apply_inverse_raw(mne.io.RawArray(pp_data, loc_info),
                                                             inv, lambda2 = 1./9., method = "dSPM", use_cps = True, pick_ori = None, verbose = VERBOSE_LVL)
        morph = mne.compute_source_morph(loc_source_data, subject_from = PATIENT_NAME,
                                         subject_to = "fsaverage", subjects_dir = ANATOMY_PATH,
                                         src_to = src_fsaverage, verbose = VERBOSE_LVL)
        morphed_src = morph.apply(loc_source_data)

def visualize(mode):
    if (mode == "skin_skull"):
        src_space = dm.load(TMP_DATA_PATH + "src_space")
        mne.viz.plot_bem(subject = PATIENT_NAME, subjects_dir = ANATOMY_PATH, src = src_space, brain_surfaces='white', orientation='coronal')
        plt.show(block = True)
    
    if (mode == "coreg"):
        info = mne.io.read_info(SENSOR_SPACE_REC_PATH, verbose = VERBOSE_LVL)
        plot_kwargs = dict(subject = PATIENT_NAME, subjects_dir = ANATOMY_PATH, surfaces = "head-dense", dig = True, eeg = [], meg = 'sensors', show_axes = True, coord_frame = 'meg')
        coreg = mne.coreg.Coregistration(info, PATIENT_NAME, ANATOMY_PATH, fiducials = "auto")
        coreg.fit_fiducials()
        coreg.fit_icp(n_iterations = 100, nasion_weight = 2., verbose = VERBOSE_LVL)
        print(np.mean(coreg.compute_dig_mri_distances()))
        coreg.omit_head_shape_points(distance = 5. / 1000)
        coreg.fit_icp(n_iterations = 100, nasion_weight = 10., verbose = VERBOSE_LVL)
        print(np.mean(coreg.compute_dig_mri_distances()))
        mne.viz.plot_alignment(info, trans = coreg.trans, **plot_kwargs, verbose = VERBOSE_LVL)
        
        #####################################
        # HACK TO MAKE THE WINDOW NOT CLOSE #
        #####################################
        fig = plt.figure()
        plt.show(block = True)

def main():
    
    #pipeline("device")
    #pipeline("missing")
    #pipeline("fs")
    #pipeline("model")
    #pipeline("apply")
    
    #visualize("skin_skull")
    visualize("coreg")


main()






