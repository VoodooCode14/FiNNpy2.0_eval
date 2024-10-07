'''
Created on Jul 23, 2024

@author: voodoocode
'''

import numpy as np
import matplotlib.pyplot as plt

import finnpy.file_io.data_manager as dm

PATH = {"freesurfer" : "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/returns/freesurfer_fastsurfer_performance/",
        "fastsurfer" : "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/returns/freesurfer_fastsurfer_performance/",
        "finnpy" : "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/returns/mne_finnpy_performance/",
        "mne" : "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/returns/mne_finnpy_performance/", 
        "evecs" : "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/returns/"}

DESCRIPTOR = {"freesurfer" : "_processing_ram_",
              "fastsurfer" : "_processing_ram_",
              "finnpy" : "_memory_",
              "mne" : "_memory_"}

OFFSETS = {"freesurfer" : {1 : [2, 6137], 2 : [2, 6203], 3 : [12, 6161], 4 : [3, 6129]},
           "fastsurfer" : {1 : [4, 2859], 2 : [4, 2847], 3 : [6, 2899], 4 : [6, 2772]},
           "finnpy" : {1 : [30, 397], 2 : [33, 404], 3 : [34, 401], 4 : [33, 399]},
           "mne" : {1 : [33, 650], 2 : [24, 685], 3 : [33, 652], 4 : [33, 656]}}

MAX_RAM = {"freesurfer" : 10,
           "fastsurfer" : 10,
           "finnpy" : 20,
           "mne" : 20}

def screen(modes):
    for (mode_idx, mode) in enumerate(modes):
        for f_cnt in range(4):
            file = open(PATH[mode] + mode + DESCRIPTOR[mode] + str(f_cnt + 1) + ".txt")
            time = list()
            mem_vals = list()
            start_time = -1
            
            for line in file.readlines():
                mem_vals.append(float(line.split("\t")[1]))
                
                loc_h = float(line.split("\t")[0].split(" ")[1].split(":")[0]) * 60 * 60
                loc_m = float(line.split("\t")[0].split(" ")[1].split(":")[1]) * 60
                loc_s = float(line.split("\t")[0].split(" ")[1].split(":")[2])
                loc_time = loc_h + loc_m + loc_s
                
                if (start_time == -1):
                    start_time = loc_time
                
                time.append(loc_time - start_time)
                
            file.close()
            
            mem_vals = np.asarray(mem_vals)
            mem_vals -= np.mean(mem_vals[:2])
            mem_vals *= 64
            
            plt.plot(time, mem_vals)
            plt.suptitle(mode + "_processing_ram_" + str(f_cnt + 1) + ".txt")
            plt.show(block = True)

def mem(modes = ["freesurfer", "fastsurfer"]):
    (fig, axes) = plt.subplots(4, int(2 * 2))
    (grp_fig, grp_axes) = plt.subplots(1, 1)
    for (mode_idx, mode) in enumerate(modes):
        avg_vals = list()
        max_vals = list()
        for f_cnt in range(4):
            file = open(PATH[mode] + mode + DESCRIPTOR[mode] + str(f_cnt + 1) + ".txt")
            time = list()
            mem_vals = list()
            start_time = -1
            
            for line in file.readlines():
                mem_vals.append(float(line.split("\t")[1]))
                
                loc_h = float(line.split("\t")[0].split(" ")[1].split(":")[0]) * 60 * 60
                loc_m = float(line.split("\t")[0].split(" ")[1].split(":")[1]) * 60
                loc_s = float(line.split("\t")[0].split(" ")[1].split(":")[2])
                loc_time = loc_h + loc_m + loc_s
                
                if (start_time == -1):
                    start_time = loc_time
                
                time.append(loc_time - start_time)
                
            file.close()
            
            mem_vals = np.asarray(mem_vals)
            mem_vals -= np.mean(mem_vals[:2])
            mem_vals *= 64
            
            avg_val = np.mean(mem_vals[(OFFSETS[mode][f_cnt + 1][0] + 3):(OFFSETS[mode][f_cnt + 1][1] - 3)])
            max_val = np.max(mem_vals[(OFFSETS[mode][f_cnt + 1][0] + 3):(OFFSETS[mode][f_cnt + 1][1] - 3)])
            avg_vals.append(avg_val)
            max_vals.append(max_val)
            
            axes[f_cnt, int(0 + mode_idx * 2)].plot(time, mem_vals)
            axes[f_cnt, int(1 + mode_idx * 2)].bar([0, 1], [avg_val, max_val])
        grp_axes.bar([mode_idx-.2, mode_idx+.2], [np.mean(avg_vals), np.mean(max_vals)],
                     yerr = [np.sqrt(np.var(avg_vals)), np.sqrt(np.var(max_vals))],
                     color = ["red", "blue"], alpha = 0.5, width = .4)
    for i in range(4):
        for j in range(4):
            axes[i, j].set_ylim([0, MAX_RAM[mode]])
    plt.show(block = True)

def time(modes):
    
    plt.figure()
    
    for (mode_idx, mode) in enumerate(modes):
        file = open(PATH[mode] + mode + "_processing_time.txt", "r")
        
        segments = [list() for _ in range(5)]
        total_times = list()
        for line in file.readlines():
            line_segs = line.split("\t")
            total_times.append(0)
            for (line_seg_idx, line_seg) in enumerate(line_segs):
                segments[line_seg_idx].append(float(line_seg))
                total_times[-1] += float(line_seg)
        
        offset = 0
        for segment in segments:
            if (len(segment) == 0):
                continue
            plt.bar([mode_idx,], [np.mean(segment),], yerr = [np.sqrt(np.var(segment)),], bottom = offset)
            offset += np.mean(segment)
        
        print(np.mean(total_times), np.sqrt(np.var(total_times)))
        
        file.close()
    plt.show(block = True)

def evecs():
    cov16 = dm.load(PATH["evecs"] + "cov16")
    cov32 = dm.load(PATH["evecs"] + "cov32")
    cov64 = dm.load(PATH["evecs"] + "cov64")
    cov128 = dm.load(PATH["evecs"] + "cov128")
    cov256 = dm.load(PATH["evecs"] + "cov256")
    covs = [cov16, cov32, cov64, cov128]
    
    
    loc_idx = -1
          
    diff = np.zeros((10, 4))
    for eval_idx in range(9):
        loc_idx = (-1 - eval_idx)
        for cov_idx in range(len(covs)):
            if (np.sign(cov256.evals[loc_idx]) == np.sign(covs[cov_idx].evals[loc_idx])):
                diff[eval_idx, cov_idx] += np.abs(cov256.evecs[loc_idx, :] - covs[cov_idx].evecs[loc_idx, :]).sum()
            else:
                diff[eval_idx, cov_idx] += np.abs(cov256.evecs[loc_idx, :] + covs[cov_idx].evecs[loc_idx, :]).sum()
    
    for eval_idx in range(0,  len(cov256.evals) - 9):
        for cov_idx in range(len(covs)):
            if (np.sign(cov256.evals[eval_idx]) == np.sign(covs[cov_idx].evals[eval_idx])):
                diff[9, cov_idx] += np.abs(cov256.evecs[loc_idx, :] - covs[cov_idx].evecs[loc_idx, :]).sum()
            else:
                diff[9, cov_idx] += np.abs(cov256.evecs[loc_idx, :] + covs[cov_idx].evecs[loc_idx, :]).sum()
    
    diff[9, :] /= len(cov256.evals) - 9            
    #===========================================================================
    # diff = np.log(diff)
    #===========================================================================
                
    plt.bar(np.arange(0, 10) - .30, diff[:, 0], width = .15)
    plt.bar(np.arange(0, 10) - .10, diff[:, 1], width = .15)
    plt.bar(np.arange(0, 10) + .10, diff[:, 2], width = .15)
    plt.bar(np.arange(0, 10) + .30, diff[:, 3], width = .15)
        
    plt.figure()
    plt.bar(np.arange(0, 9) - .30, diff[:9, 0], width = .15)
    plt.bar(np.arange(0, 9) - .10, diff[:9, 1], width = .15)
    plt.bar(np.arange(0, 9) + .10, diff[:9, 2], width = .15)
    plt.bar(np.arange(0, 9) + .30, diff[:9, 3], width = .15)
    
    plt.show(block = True)

TMP_DATA_PATH = "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/data/finnpy/tmp/"
SENSOR_SPACE_REC_PATH = "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/data/finnpy/raw_data/demo_patient/meg/rec.fif"
import finnpy.src_rec.inv_mdl
import mne
import scipy.spatial.distance

#===============================================================================
# def evecs2():
#     cov16 = dm.load(PATH["evecs"] + "cov16")
#     cov32 = dm.load(PATH["evecs"] + "cov32")
#     cov64 = dm.load(PATH["evecs"] + "cov64")
#     cov128 = dm.load(PATH["evecs"] + "cov128")
#     cov256 = dm.load(PATH["evecs"] + "cov256")    
#     
#     vals = [np.mean(np.abs(cov16.evecs / cov256.evecs)), np.mean(np.abs(cov32.evecs / cov256.evecs)),
#             np.mean(np.abs(cov64.evecs / cov256.evecs)), np.mean(np.abs(cov128.evecs / cov256.evecs)), 
#             np.mean(np.abs(cov256.evecs / cov256.evecs))]
#     vars = [np.mean(np.abs(cov16.evecs / cov256.evecs)), np.mean(np.abs(cov32.evecs / cov256.evecs)),
#             np.mean(np.abs(cov64.evecs / cov256.evecs)), np.mean(np.abs(cov128.evecs / cov256.evecs)), 
#             np.mean(np.abs(cov256.evecs / cov256.evecs))]
#     vals = np.asarray(vals)
#     vars = np.asarray(vars)
#     
#     vals = np.log10(vals)
#     vars = np.log10(vars)
#     
#     plt.bar(np.arange(0, 5), vals, yerr = np.abs(vars))
#     plt.show(block = True)
#===============================================================================
    
def evecs2():
    cov016 = dm.load(PATH["evecs"] + "cov16")
    cov032 = dm.load(PATH["evecs"] + "cov32")
    cov064 = dm.load(PATH["evecs"] + "cov64")
    cov128 = dm.load(PATH["evecs"] + "cov128")
    cov256 = dm.load(PATH["evecs"] + "cov256")
    
    vector_axis = 0
    
    #Euclidean distance
    vals_016 = np.sqrt(np.sum(np.power(cov016.evecs - cov256.evecs, 2), axis = vector_axis))
    vals_032 = np.sqrt(np.sum(np.power(cov032.evecs - cov256.evecs, 2), axis = vector_axis))
    vals_064 = np.sqrt(np.sum(np.power(cov064.evecs - cov256.evecs, 2), axis = vector_axis))
    vals_128 = np.sqrt(np.sum(np.power(cov128.evecs - cov256.evecs, 2), axis = vector_axis))
    
    #Weighing by eigenvalues
    vals_016 = vals_016 * cov256.evals
    vals_032 = vals_032 * cov256.evals
    vals_064 = vals_064 * cov256.evals
    vals_128 = vals_128 * cov256.evals
    
    vars_016 = np.sqrt(np.var(vals_016))
    vars_032 = np.sqrt(np.var(vals_032))
    vars_064 = np.sqrt(np.var(vals_064))
    vars_128 = np.sqrt(np.var(vals_128))
    
    vals_016 = np.sum(vals_016)
    vals_032 = np.sum(vals_032)
    vals_064 = np.sum(vals_064)
    vals_128 = np.sum(vals_128)
    
    vars = [vars_016, vars_032, vars_064, vars_128]
    vars = np.asarray(vars)
    vals = [vals_016, vals_032, vals_064, vals_128]
    vals = np.asarray(vals)
    
    vars /= vals[-1]
    vals /= vals[-1]
    
    #===========================================================================
    # vals = np.log10(vals)
    #===========================================================================

    plt.bar(np.arange(0, 4), vals, yerr = np.abs(vars))
    plt.show(block = True)

def evecs3():
    cov016 = dm.load(PATH["evecs"] + "cov16")
    cov032 = dm.load(PATH["evecs"] + "cov32")
    cov064 = dm.load(PATH["evecs"] + "cov64")
    cov128 = dm.load(PATH["evecs"] + "cov128")
    cov256 = dm.load(PATH["evecs"] + "cov256")
    
    vector_axis = 0
    
    #Euclidean distance
    vals_016 = np.sqrt(np.sum(np.power(cov016.evecs - cov256.evecs, 2), axis = vector_axis))
    vals_032 = np.sqrt(np.sum(np.power(cov032.evecs - cov256.evecs, 2), axis = vector_axis))
    vals_064 = np.sqrt(np.sum(np.power(cov064.evecs - cov256.evecs, 2), axis = vector_axis))
    vals_128 = np.sqrt(np.sum(np.power(cov128.evecs - cov256.evecs, 2), axis = vector_axis))
    
    #Weighing by eigenvalues
    vals_016 = vals_016 * cov256.evals
    vals_032 = vals_032 * cov256.evals
    vals_064 = vals_064 * cov256.evals
    vals_128 = vals_128 * cov256.evals
    
    vars_016 = np.sqrt(np.var(vals_016))
    vars_032 = np.sqrt(np.var(vals_032))
    vars_064 = np.sqrt(np.var(vals_064))
    vars_128 = np.sqrt(np.var(vals_128))
    
    vals_016 = np.sum(vals_016)
    vals_032 = np.sum(vals_032)
    vals_064 = np.sum(vals_064)
    vals_128 = np.sum(vals_128)
    
    vars = [vars_016, vars_032, vars_064, vars_128]
    vars = np.asarray(vars)
    vals = [vals_016, vals_032, vals_064, vals_128]
    vals = np.asarray(vals)
    
    #===========================================================================
    # vars /= vals[-1]
    # vals /= vals[-1]
    #===========================================================================
    
    #===========================================================================
    # vals = np.log10(vals)
    #===========================================================================

    plt.bar(np.arange(0, 4), vals, yerr = np.abs(vars))
    plt.show(block = True)

#===============================================================================
# #screen(["finnpy", "mne"]) 
#        
# #mem(["fastsurfer", "freesurfer"])
# #mem(["finnpy", "mne"])
# #time(["fastsurfer", "freesurfer"])
# #time(["finnpy", "mne"])
# 
# #evecs()
# #evecs2()
#===============================================================================


#===============================================================================
# time(["fastsurfer", "freesurfer"])
#===============================================================================

#===============================================================================
# time(["finnpy", "mne"])
#===============================================================================

#===============================================================================
# mem(["fastsurfer", "freesurfer"])
#===============================================================================

#===============================================================================
# mem(["finnpy", "mne"])
#===============================================================================

evecs2()

#===============================================================================
# evecs3()
#===============================================================================
