'''
Created on Oct 10, 2024

@author: voodoocode
'''

import psutil
import time

file = open("/home/voodoocode/Downloads/tmp11/test.txt", "w")

while (True):
    file.write(str(psutil.virtual_memory()[3]) + "\n")
    file.flush()
    time.sleep(1)

print()
