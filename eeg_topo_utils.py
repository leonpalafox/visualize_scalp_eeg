'''
Utilities to create plots of topographies of the eeg, they require
the mvpa2 library 
Which usually is installed form ubuntu:
sudo apt-get install python-mvpa2
Also, the latestes version has some compatibility issues with iPython
so you have to change some code in the source library
If you don't use iPython, though, eveyrthing is fine

Many of the features are hard coded because of the static mapping to
the topography map
'''
from mvpa2.suite import *
import numpy as np
import pylab as plt
import utils
import wutils as wu
#MAPEEG is a variable that helps us map the 64 electrode default setting in mvpa to 64 electrodes
MAPEEG=['Fp1', 'Fpz', 'Fp2','AF7','AF3','AF4','AF8','F7','F5','F3','F1',
        'Fz','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCz', 
        'FC2','FC4','FC6','FT8','T7','C5','C3','C1','Cz','C2',
        'C4','C6','T8','TP7','CP5','CP3','CP1','CPz','CP2','CP4',
        'CP6','TP8','P7','P5','P3','P1','Pz','P2','P4','P6',
        'P8','PO7','PO3','POz','PO4','PO8','O1','Oz','O2','TP9','TP10',False,False]
sensors=XAVRSensorLocations(os.path.join(pymvpa_dataroot,'xavr1020.dat'))

def plot_top_clf(clf, clf_features, clf_title):
    '''
    This script uses a classifier as an input, for the case when our result 
    coomes from a classification algorithm
    '''
    electrode_coeffs = np.zeros(64)
    electrod_topography = np.zeros(len(sensors.names))
    if clf_features == 'Single Channel':
        electrode_coeffs[:clf.coef_.shape[1]] = clf.coef_
    if clf_features == 'Large Laplacian':
        print clf.coef_[0].shape
        print electrode_coeffs[:,(27,28,29,30,31,36,37,38,39,40)].shape
        electrode_coeffs[:,(27,28,29,30,31,36,37,38,39,40)] = clf.coef_[0]
    if clf_features == 'Small Laplacian':
        electrode_coeffs[17:23] = clf.coef_[0:7]
        electrode_coeffs[26:32] = clf.coef_[7:14]
        electrode_coeffs[35:42] = clf.coef_[14:21]
    for sens_idx, sens_name in enumerate(MAPEEG):
        if sens_name:
            electrod_topography[sensors.names.index(sens_name)]=electrode_coeffs[sens_idx]
    plot_head_topography(electrod_topography,sensors.locations())
    plt.colorbar()
    plt.title(clf_title, fontsize = 24)

def plot_top_array(input_feat, clf_features, clf_title):
    '''
    This script uses an as an input,
    The array is treated as a 1x#feature vector
    '''
    if len(input_feat.shape) < 2:
        input_feat = input_feat[None, :]
    electrode_coeffs = np.zeros(64)
    electrod_topography = np.zeros(len(sensors.names))
    if clf_features == 'Single Channel':
        electrode_coeffs[:input_feat.shape[1]] = input_feat
    if clf_features == 'Large Laplacian':
        print input_feat[0].shape
        print electrode_coeffs[:,(27,28,29,30,31,36,37,38,39,40)].shape
        electrode_coeffs[:,(27,28,29,30,31,36,37,38,39,40)] = input_feat[0]
    if clf_features == 'Small Laplacian':
        electrode_coeffs[17:23] = input_feat[0:7]
        electrode_coeffs[26:32] = input_feat[7:14]
        electrode_coeffs[35:42] = input_feat[14:21]
    for sens_idx, sens_name in enumerate(MAPEEG):
        if sens_name:
            electrod_topography[sensors.names.index(sens_name)]=electrode_coeffs[sens_idx]
    #plt.figure()
    plot_head_topography(electrod_topography,sensors.locations())
    plt.colorbar()
    plt.title(clf_title, fontsize = 24)
    
def plot_average_features_head(data_test, labels_test, clf, clf_features, feat_title):
    ''' This creates 2 subplots that have the plots of the head and 
    The averaged time series'''
    cue_idx=utils.getIndex(labels_test,'up') ###Getting the indexes for the cue onset
    left_length = 10.0
    right_length = 15.0
    labels_struct, _ = utils.getWindows(labels_test,left_length, right_length, cue_idx)
    data_struct, time_axis = utils.getWindows(data_test,left_length, right_length, cue_idx)
    plt.subplot2grid((2,9), (0,0), colspan=5)
    plot_data = data_struct.mean(2)
    factor = plot_data.shape[0]/5
    p1,=plt.plot(time_axis, plot_data.mean(1), label='Mean Values', color = 'black', linewidth=2.5, marker = 'x')
    scaler = plot_data.mean(1).max()
    wu.plot_shadowed_data_time(time_axis, scaler*labels_struct[:,:,0],'blue', alpha_value=0.2)
    p2 = plt.Rectangle((0, 0), 1, 1, fc="0.75",alpha=0.5)
    plt.legend()
    plt.xlabel('Time[s]', fontsize = 16)
    plt.ylabel('Average ' + feat_title + 'over channels', fontsize = 24)
    plt.title('Averaged Superimposed '+ feat_title + ' over channels', fontsize = 24)
    plt.xlim([-10,15])
    plt.grid()
    feat_title = ''
    plt.subplot2grid((2,9), (1,0), colspan=1)
    plot_top_array(plot_data[0,:50],'Single Channel', feat_title)
    plt.subplot2grid((2,9), (1,1), colspan=1)
    plot_top_array(plot_data[factor,:50],'Single Channel', feat_title)
    plt.subplot2grid((2,9), (1,2), colspan=1)
    plot_top_array(plot_data[factor*2,:50],'Single Channel', feat_title)
    plt.subplot2grid((2,9), (1,3), colspan=1)
    plot_top_array(plot_data[factor*3, :50],'Single Channel', feat_title)
    plt.subplot2grid((2,9), (1,4), colspan=1)
    plot_top_array(plot_data[factor*4, :50],'Single Channel', feat_title)
    plt.subplot2grid((2,9), (0,5), colspan=4, rowspan = 2)
    feat_title = 'Derivatives'
    plot_top_clf(clf, clf_features, feat_title + ' Feature Weights')
    #ax2 = plt.subplot2grid((3,3), (1,0), colspan=2)
    #ax3 = plt.subplot2grid((3,3), (1, 2), rowspan=2)
    #ax4 = plt.subplot2grid((3,3), (2, 0))
    #ax5 = plt.subplot2grid((3,3), (2, 1))
