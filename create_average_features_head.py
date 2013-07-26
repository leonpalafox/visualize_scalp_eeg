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
pickle_filename = 'pickle_class_8_12_large_lapl.p'
clf_dict = pickle.load(open(pickle_filename, "rb"))
clf = clf_dict['Derivatives']
clf_features = 'Large Laplacian'
###############################################################################
file_list = ['/features.csv', '/deriv_features.csv']
baseline_labels, prehold_labels, hold_labels, cue_labels_data, data_, deriv_data = wu.create_dataset(number_of_experiments, 
                                                                        test_subject, experiment_months, file_list)
data_= wu.normalize_data(data_)
feat_title = 'Derivative'
tutils.plot_average_features_head(deriv_data, cue_labels_data, clf, clf_features, feat_title)
sys.exit()
