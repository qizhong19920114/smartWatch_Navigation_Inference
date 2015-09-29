import kMeans
import ProbIN
from numpy import *
import subprocess
import numpy as np

datMat = mat(kMeans.loadDataSet('motionData_Training.txt'))
kMeans.biKmeans(datMat,12)

# datMat2 = mat(kMeans.loadDataSet('GPS_1Hz_training.txt'))
# kMeans.biKmeans(datMat2,7)