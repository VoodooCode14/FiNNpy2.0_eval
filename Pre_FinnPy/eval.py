'''
Created on Jul 23, 2024

@author: voodoocode
'''

import numpy as np
import matplotlib.pyplot as plt

import finnpy.file_io.data_manager as dm

PATH = {"freesurfer" : "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/returns/freesurfer_fastsurfer_performance/",
        "fastsurfer" : "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/returns/freesurfer_fastsurfer_performance/",
        "finnpy" : "/home/voodoocode/Downloads/tmp11/stats/finnpy/",
        "mne" : "/home/voodoocode/Downloads/tmp11/stats/mne/", 
        "evecs" : "/home/voodoocode/Desktop/stuff/FinnPy_2/tmp/returns/"}

MEM_STARTS = {"finnpy" : [10, 90, 170, 250, 330],
              "mne" : [18, 215, 415, 615, 812]}

    
def evecs():
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

    plt.bar(np.arange(0, 4), vals, yerr = np.abs(vars))
    plt.show(block = True)

def screen_mem(mode):
    vals = [list(), list()]
    for idx in range(2):
        file = open(PATH[mode[idx]] + "memory.txt")
        
        vals[idx] = list()
        while (True): 
            line = file.readline()
            if (len(line) == 0):
                break
            vals[idx].append(int(line))
    (fig, axes) = plt.subplots(2, 1)
    axes[0].plot(vals[0])
    axes[1].plot(vals[1])
    
    plt.show(block = True)

def eval_mem(mode):
    vals = [list(), list()]
    ref_vals = [list(), list()]; mean_vals = [list(), list()]; max_vals = [list(), list()]
    for idx in range(2):
        file = open(PATH[mode[idx]] + "memory.txt")
        
        vals[idx] = list()
        while (True): 
            line = file.readline()
            if (len(line) == 0):
                break
            vals[idx].append(int(line))
        for ref_idx in range(len(MEM_STARTS[mode[idx]]) - 1):
            ref_vals[idx].append(np.mean(np.asarray(vals[idx])[np.asarray(MEM_STARTS[mode[idx]])[[ref_idx, ref_idx + 1]]]))
            mean_vals[idx].append(np.mean(np.asarray(vals[idx])[MEM_STARTS[mode[idx]][ref_idx]:MEM_STARTS[mode[idx]][ref_idx + 1]]))
            max_vals[idx].append(np.max(np.asarray(vals[idx])[MEM_STARTS[mode[idx]][ref_idx]:MEM_STARTS[mode[idx]][ref_idx + 1]]))
            
            mean_vals[idx][-1] -= ref_vals[idx][-1]
            max_vals[idx][-1] -= ref_vals[idx][-1]
        
        mean_vals[idx] /= np.power(1024, 3)
        max_vals[idx] /= np.power(1024, 3)
        
        max_vals[idx] /= np.mean(mean_vals[0])
        mean_vals[idx] /= np.mean(mean_vals[0])
        
        print(np.mean(mean_vals[idx]), np.min(mean_vals[idx]), np.max(mean_vals[idx]))
        print(np.mean(max_vals[idx]), np.min(max_vals[idx]), np.max(max_vals[idx]))
        print()

def eval_speed():
    mne_times = np.asarray([186973797090, 188709828191, 190244675808, 186572477105])
    finnpy_times = np.asarray([70339662700, 69824675631, 70017337506, 70001017578])
    
    mne_times = mne_times/(np.power(1000., 3) * 60)
    finnpy_times = finnpy_times/(np.power(1000., 3) * 60)
    
    finnpy_times = finnpy_times / np.mean(mne_times)
    mne_times = mne_times / np.mean(mne_times)
    
    print(np.mean(mne_times), np.min(mne_times), np.max(mne_times))
    print(np.mean(finnpy_times), np.min(finnpy_times), np.max(finnpy_times))

#screen_mem(["finnpy", "mne"])
#eval_mem(["finnpy", "mne"])
eval_speed()
#evecs()

