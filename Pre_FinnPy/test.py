
'''
Created on Jul 14, 2024

@author: voodoocode
'''

import mne
import finnpy.src_rec.sen_cov
import mpmath
import mpmath.libmp
import gmpy2
import time
import numpy as np

raw_file = mne.io.read_raw_fif("/home/voodoocode/Downloads/tmp/data/finnpy/empty_room/rec.fif", preload = True, verbose = "ERROR")
(bio_sensor_noise_cov, ch_names) = finnpy.src_rec.sen_cov._calc_sensor_noise_cov(raw_file, method = None, method_params = None)

print(mpmath.libmp.BACKEND)

start = time.time()
(eigen_val, eigen_vec) = np.linalg.eigh(bio_sensor_noise_cov)
print(time.time() - start)
start = time.time()
bio_sensor_noise_cov = np.asarray(bio_sensor_noise_cov, dtype = np.float128)
(eigen_val, eigen_vec) = np.linalg.eigh(bio_sensor_noise_cov)
print(time.time() - start)
start = time.time()
mat = mpmath.matrix(bio_sensor_noise_cov)
mat.ctx.dps = 40
(eigen_val, eigen_vec) = mpmath.eigsy(mat)
print(time.time() - start)






