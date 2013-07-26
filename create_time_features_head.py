#Load libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import cross_validation
import pylab as plt
import wutils as wu #Has the libraries to do windowing related techniques
import var_file as vf #Has all the definition of the different used variables
import utils
import sys, os
import eeg_topo_utils as tutils
from time import sleep
################################################################################
#################################################################################
########################################################################3
################Variables rleated to the datafiles##############################
number_of_experiments=10
subjects=['Krish','Josh']#List all the subjects names for the file format
test_subject=subjects[1]#Select the subject for this current set of plots
experiment_days=np.arange(12,32)#days of the month for different runs
experiment_months=['June']#months of the experiments
samp_frequency=vf.get_param_value('sampling_frequency')
#############################################################################
########################################################################
file_list = ['/features.csv', '/deriv_features.csv']
baseline_labels, prehold_labels, hold_labels, cue_labels_data, data_, deriv_data = wu.create_dataset(number_of_experiments, 
                                                                        test_subject, experiment_months, file_list)
data_= wu.normalize_data(data_)

for i in range(2000,4000):
    plt.figure()
    plt.subplots_adjust( hspace=0.4 )
    plt.subplot(2,1,1)
    wu.plot_raw_data(data_[0:(i+1),27:29], samp_frequency, 'Time[sec]', 'Power', 'Raw data')
    plt.subplot(2,1,2)
    pass_data_ = data_[i,:62]
    pass_data_[[60,61]] = [0, 0.5] 
    tutils.plot_top_array(pass_data_, 'Single Channel', 'Power')
    plt.savefig('movie/movie_file_'+str(i))
    plt.close()
sys.exit()
