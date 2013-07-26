''''
This library has utilities useful for the windowing functions
'''
import os
import itertools
import numpy as np
import utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import svm, metrics
import GlobalEM as gem
import pylab as plt
import var_file as vf #Has all the definition of the different used variables

def preprocess_data(data, window_length):
    """This arranges the data so its consistent with the window length"""
    if len(data.shape)<2:
        data = data[None, :]
    time, dims = data.shape
    while (time%window_length)!=0:
        data = np.delete(data, -1, 0) 
        time, dims = data.shape
    return data
    
def create_windows(data, window_length):
    """This creates windowed data given a length"""
    #Output structure has [window_length,Dims,Windows]
    if len(data.shape)<2:
        data = data[None, :]
    time, dims = np.shape(data)
    number_window = int(time/window_length)
    #Preprocess makes the data consistent with the window size
    data = preprocess_data(data, window_length)
    data_structre = np.zeros((window_length, dims, number_window))
    for window_idx in range(number_window):
        data_structre[:,:,window_idx] = \
        data[window_idx*window_length:(window_idx+1)*window_length, :]
    return data_structre

def create_labels(start_indexes, end_indexes, dataset):
    time=dataset.shape[0]
    flip_flop_labels=sorted(start_indexes+end_indexes)
    #obtains all the filters, is going t work as a flipflop
    if flip_flop_labels[0]==start_indexes[0]:
        new_labels=np.zeros((time,1))
    else:
        new_labels=np.ones((time,1))
    for labels_idx in flip_flop_labels:
        new_labels[labels_idx:,0]=1-new_labels[labels_idx:,0]
    return new_labels

def create_windows_baseline(data, window_length, join_indexes):
    """This creates windowed data given a length and a series of indexes to avoid """
    #The indexes are indicators of concatenated data
    #Output structure has [window_length,Dims,Windows]
    time, dims = np.shape(data)
    non_windowed=len(join_indexes)
    number_window = int(time/window_length)-non_windowed
    data_structre = np.zeros((window_length, dims, number_window))
    for window_idx in range(number_window):
        if np.any(join_indexes>window_idx*window_length) and np.any(join_indexes<(window_idx+1)*window_length):
            pass
            del join_indexes[0]
        else:
            data_structre[:,:,window_idx] = \
            data[window_idx*window_length:(window_idx+1)*window_length, :]
    #The returned data does not have as many windows as it should
    return data_structre

def svm_format(data, labels):
    '''This function creates anSVM format compiant file from a 3D windowed set'''
    if data.shape[0]!=labels.shape[0] or data.shape[2]!=labels.shape[2]:
        print 'labels and data are not the same'
        return
    else:
        time, dims, trials = data.shape
        svm_data=np.zeros((trials, dims))
        svm_labels=np.zeros((trials,1))
        for trial_idx in range(trials):
            svm_data[trial_idx, :] = data[:, :, trial_idx].mean(0)
            svm_labels[trial_idx, :] = np.round(labels[:, :, trial_idx].mean(0))
        return svm_data, svm_labels

def score_plot_svm(clf, data_test, label_test):
    predicted=clf.predict(data_test)
    probas_=clf.predict_proba(data_test)
    sampling_frequency=vf.get_param_value('sampling_frequency')
    plot_raw_data(predicted[:, None], sampling_frequency, 'Time [sec]', 'Probability', 'Detection Probability')
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(label_test,probas_[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
def score_plot_svm_shaded(clf_list, data_test, label_test, hold_labels_test, cue_labels_test, baseline_labels, prehold_labels, data_list, title_list):
    color_list = ['orange', 'cyan', 'magenta', 'green']
    numb_features=data_test.shape[1]/2
    sampling_frequency=vf.get_param_value('sampling_frequency')
    offset = 1.0
    plot_data=-5+10*normalize_data(data_test)
    plot_raw_data(plot_data, sampling_frequency, 'Time [sec]', 'Normalized Derivative', 'Tracking Prehold + Hold')
    scaling=plot_data.max()+2
    plot_shadowed_data_non_zero(scaling+(1-hold_labels_test), 'red')
    plot_shadowed_data(scaling*label_test,'blue', alpha_value=0.2)
    for clf_idx, bin_clf in enumerate(clf_list):
        predicted_ = bin_clf.predict(data_list[clf_idx])
        plot_shadowed_data_new_class(scaling-offset+(1-0.5*predicted_[:,None]),color_list[clf_idx],
                                    title_list[clf_idx])
        offset += 0.5 
    
    plot_shadowed_data_new_class(scaling-offset+(1-0.5*hold_labels_test),'blue','Threshold')
    plot_shadowed_data_non_zero(scaling+(1-cue_labels_test), 'green')
    plot_shadowed_data_non_zero(scaling+(1-baseline_labels), 'black')
    plot_shadowed_data_non_zero(scaling+(1-prehold_labels), 'blue')
    plt.legend(loc=4,prop={'size':16})
    
    
def score_plot_svm_shaded_single(clf_list, data_test, label_test, hold_labels_test, cue_labels_test, baseline_labels, prehold_labels):
    plt.figure()
    numb_features=data_test.shape[1]/2
    predicted_deriv=clf_list[0].predict(data_test)
    sampling_frequency=vf.get_param_value('sampling_frequency')
    plot_data=-5+10*normalize_data(data_test)
    plot_raw_data(plot_data, sampling_frequency, 'Time [sec]', 'Normalized Derivative', 'Tracking Prehold + Hold')
    scaling=plot_data.max()+2
    plot_shadowed_data_non_zero(scaling+(1-hold_labels_test), 'red')
    plot_shadowed_data(scaling*label_test,'blue', alpha_value=0.2)
    plot_shadowed_data_new_class(scaling-1+(1-0.5*predicted_deriv[:,None]),'orange','Power + Derivatives')
    plot_shadowed_data_new_class(scaling-2.5+(1-0.5*hold_labels_test),'blue','Threshold')
    plot_shadowed_data_non_zero(scaling+(1-cue_labels_test), 'green')
    plot_shadowed_data_non_zero(scaling+(1-baseline_labels), 'black')
    plot_shadowed_data_non_zero(scaling+(1-prehold_labels), 'blue')
    plt.legend(loc=4,prop={'size':16})
def plot_histogram(clf_list, data_test, hold_labels_test, labels_test, title_list, algorithm):
    ''' This creates the histograms using a generic data set and set of labels'''
    ''' The histogram only detects first onset, and ignores subsequent ones '''
    cue_idx=utils.getIndex(labels_test,'up') ###Getting the indexes for the cue onset
    left_length = 2.0
    right_length = 5.0
    plt.figure()
    plt.suptitle('Tracking Hold + Prehold ('+ algorithm +')', fontsize = 14)
    for clf_idx in range(len(clf_list)):
        plt.subplot(4,1,clf_idx+1)
        pred_data = clf_list[clf_idx].predict(data_test[clf_idx])
        labels_struct, _ = utils.getWindows(labels_test,left_length, right_length, cue_idx)
        hold_labels_struct, _ = utils.getWindows(hold_labels_test,left_length, right_length, cue_idx)
        pred_data_struct, time_axis = utils.getWindows(pred_data,left_length, right_length, cue_idx)
        first_pass_struct = utils.detect_first_pass(pred_data_struct)
        #stem_plot = plt.stem(time_axis, first_pass_struct.sum(2))
        cdf = np.cumsum(first_pass_struct.sum(2))
        p1, = plt.plot(time_axis, cdf/np.max(cdf), 'red', label='Cumulative Distribution', linewidth=2.5, linestyle="-")
        plot_shadowed_data_time(time_axis, hold_labels_struct[:,:,0],'blue', alpha_value=0.2)
        #stem_depth_array(time_axis, first_pass_struct)
        plt.xlabel('Time', fontsize = 14)
        plt.ylabel('Cue', fontsize = 14)
        plt.title(title_list[clf_idx])
        p2=plt.Rectangle((0, 0), 1, 1, fc="blue",alpha=0.2)
        plt.legend([p1, p2],['Cumulative Distribution', 'Threshold Detection'], loc=4)
        plt.grid()

def plot_histogram_iter(clf_list, data_test, hold_labels_test, labels_test, title_list, plot_label, color, marker):
    ''' This creates the histograms using a generic data set and set of labels'''
    ''' The histogram only detects first onset, and ignores subsequent ones '''
    ''' It works for the case of iterative data, so it superimposes from previous histograms'''
    cue_idx=utils.getIndex(labels_test,'up') ###Getting the indexes for the cue onset
    left_length = 2.0
    right_length = 5.0

    for clf_idx in range(len(clf_list)):
        plt.subplot(3,1,clf_idx+1)
        pred_data = clf_list[clf_idx].predict(data_test[clf_idx])
        labels_struct, _ = utils.getWindows(labels_test,left_length, right_length, cue_idx)
        hold_labels_struct, _ = utils.getWindows(hold_labels_test,left_length, right_length, cue_idx)
        pred_data_struct, time_axis = utils.getWindows(pred_data,left_length, right_length, cue_idx)
        first_pass_struct = utils.detect_first_pass(pred_data_struct)
        #stem_plot = plt.stem(time_axis, first_pass_struct.sum(2))
        cdf = np.cumsum(first_pass_struct.sum(2))
        p1, = plt.plot(time_axis, cdf/np.max(cdf),ls='-',marker = marker, color = color, label=plot_label, linewidth=2.5)
        plt.grid(True)
        if plot_label == '1':
            plot_shadowed_data_time(time_axis, hold_labels_struct[:,:,0],'blue', alpha_value=0.2)
            #stem_depth_array(time_axis, first_pass_struct)
            plt.xlabel('Time', fontsize = 14)
            plt.ylabel('Cue', fontsize = 14)
            plt.title(title_list[clf_idx])
            p2=plt.Rectangle((0, 0), 1, 1, fc="blue",alpha=0.2)
            if clf_idx == 1:
                plt.legend([p1, p2],[plot_label, 'Threshold Detection'], loc=4)
        else:
            if clf_idx == 1:
                plt.legend(loc=4)

def plot_average_feature_subplots(data_list, ylist, labels, algorithm):
    plt.figure()
    for data_idx, data_ in enumerate(data_list):
        plt.subplot(1, len(data_list), data_idx+1)
        plot_average_feature(data_, labels)
        plt.xlabel('Time[s]', fontsize = 16)
        plt.ylabel(ylist[data_idx], fontsize = 16)
        plt.title('Mean '+ylist[data_idx]+' for '+algorithm, fontsize = 16)
        plt.grid(True)

def plot_average_feature(data_test, labels_test):
    '''Plots superimposed features'''
    '''The mean and the 95% confidence interval'''
    cue_idx=utils.getIndex(labels_test,'up') ###Getting the indexes for the cue onset
    left_length = 2.0
    right_length = 5.0
    labels_struct, _ = utils.getWindows(labels_test,left_length, right_length, cue_idx)
    data_struct, time_axis = utils.getWindows(data_test,left_length, right_length, cue_idx)
    plot_data = data_struct.mean(2)
    data_std = data_struct.std(2)
    sigma = 1.96*data_std/data_struct.shape[2]
    upper_bound = plot_data + sigma
    lower_bound = plot_data - sigma
    print time_axis.shape
    scaler = plot_data.mean(1).max()
    p1,=plt.plot(time_axis, plot_data.mean(1), label='Mean Values', color = 'black', linewidth=2.5, marker = 'x')
    plt.fill_between(time_axis, lower_bound.mean(1), upper_bound.mean(1), color = '0.75',alpha = 0.5)
    plot_shadowed_data_time(time_axis, scaler*labels_struct[:,:,0],'blue', alpha_value=0.2)
    p2 = plt.Rectangle((0, 0), 1, 1, fc="0.75",alpha=0.5)
    plt.legend([p1, p2],['Mean', '95 % CI'], loc=4)
    
def plot_depth_array(time, data):
    '''Plots a 3D array over time'''
    ntp, dims, trials = np.shape(data)
    for trial_idx in range(trials):
        plt.plot(time, data[:, :, trial_idx])

def stem_depth_array(time, data):
    '''Plots a 3D array over time'''
    ntp, dims, trials = np.shape(data)
    for trial_idx in range(trials):
        print data.shape
        plt.stem(time, data[:, :, trial_idx])
        
def get_data_differences(read_data):
    '''Finds every possible differences between the columns of the read_data 
       read_data: NxM variable where M are the features
       returns diff_data, and NxR variables
       R is the number of every possible combination of 2 columns'''
    if len(read_data.shape) != 2:
        print 'The data format is not consistent'
    _, data_columns = read_data.shape
    data_difference = np.diff(read_data[:, list(itertools.combinations(range(data_columns), 2))])[..., 0]
    #This indexes every combination and calculates the diffetence
    return data_difference
    
def normalize_data(data_structure):
    """This takes the data structure and normalizes each segment form 0-1"""
    min_max_scaler = MinMaxScaler()
    data_structure = \
            min_max_scaler.fit_transform(data_structure)
    return data_structure

def create_filters(data_structure, n_states, em_iter):
    ''' This function takes the windowed pieces and train klaman filters'''
    ''' It returns sets of parameters for later clustering '''
    time_window, dims, windows=data_structure.shape
    filter_array=[]
    for wind_idx in range(windows):
        filter_array.append(gem.generate_linear_model_single(data_structure[:,:,wind_idx],n_states,em_iter))#generates parameters using a Kalman Filter
    return filter_array

def generate_dataset_for_clustering(filter_array):
    ''' This function takes as input classes of filters'''
    ''' It then arranges them in vectors for later clustering'''
    data_dims=len(filter_array)
    dim_state=filter_array[0].n_dim_state
    clustering_data_set=np.zeros((data_dims, dim_state*dim_state+dim_state))
    data_idx=0;
    for filt in filter_array:
        trans_part=np.reshape(filt.transition_matrices,dim_state*dim_state,1)
        offset_part=filt.transition_offsets
        clustering_data_set[data_idx,:]=np.concatenate((trans_part,offset_part))
        data_idx+=1
    return clustering_data_set

def plot_relevant_plots(cluster_class,data_structure):
    '''This function creates plots for every cluster'''
    '''Also plots the membership density'''
    samp_frequency=vf.get_param_value('sampling_frequency')
    time,dims,trials=data_structure.shape
    time_axis=np.arange(0,time)/float(samp_frequency)
    figu=plt.figure()
    h=np.ceil(np.sqrt(cluster_class.n_clusters))
    for fig in range(cluster_class.n_clusters):
        ax = figu.add_subplot(h,h,fig+1)
        for idx in np.nonzero(cluster_class.labels_==fig)[0]:    
            plt.plot(time_axis,data_structure[:,:,idx])
            plt.title('Cluster'+str(fig+1))
            plt.ylabel('Magnitude')
            plt.xlabel('Time[sec]')
        plt.grid()
    x_axis=np.array(list(set(cluster_class.labels_)))
    freq_points=np.bincount(cluster_class.labels_)/float(np.sum(np.bincount(cluster_class.labels_)))
    figu.subplots_adjust(hspace=1)
    plt.figure()
    plt.stem(x_axis+1,freq_points,'-')
    plt.xlabel('Cluster Label')
    plt.ylabel('Normalized Frequency')
    plt.grid()

def generate_lds_cluster(cluster_class, data_structure):
    '''This function takes the K clusters and generates'''
    '''k LDS using the standard approach'''
    n_clusters = cluster_class.n_clusters
    n_states = vf.get_param_value('EM_latent_variables')
    em_iter = vf.get_param_value('EM_Iterations')
    #Generate the training datasets
    filt_array=[]
    for cluster_idx in range(cluster_class.n_clusters):
        train_idx = np.nonzero(cluster_class.labels_ == cluster_idx)[0]
        print len(train_idx)
        if len(train_idx)>1:
            train_dataset = data_structure[:, :, train_idx]
            new_filter = gem.generate_linear_model(train_dataset, n_states,em_iter)
        else:
            train_dataset = data_structure[:, :, train_idx]
            new_filter = gem.generate_linear_model_single(train_dataset[:,:,0], n_states, em_iter)#generates parameters using a Kalman Filter
            #returns a list that contains filters, one per cluster
        filt_array.append(new_filter)
    return filt_array

def plot_cluster_lds_samples(lds_list,window_length):
    ''' Plots samples from the respective filters for each cluster'''
    samp_frequency = vf.get_param_value('sampling_frequency')
    n_samples=10
    time = window_length
    time_axis = np.arange(0,time)/float(samp_frequency)
    filt_idx = 1
    h = np.ceil(np.sqrt(len(lds_list)))
    figu = plt.figure()
    for filter_idx in lds_list:
        ax = figu.add_subplot(h,h,filt_idx)
        for i in range(n_samples):
            plt.plot(time_axis, filter_idx.sample(int(window_length))[1])
        plt.ylabel('Magnitude')
        plt.xlabel('Time')
        plt.title('Samples from Filter '+ str(filt_idx))
        filt_idx += 1
        plt.grid()

def plot_raw_data(data, sampling_frequency, x_label, y_label, title):
    '''This function takes as an input a Time x Dims dataset'''
    time, dims = data.shape
    time_axis = np.arange(0,time)/float(sampling_frequency)
    plt.plot(time_axis, data)
    plt.xlabel(x_label, fontsize = 16)
    plt.ylabel(y_label, fontsize = 16)
    plt.title(title, fontsize = 16)
    plt.grid()

def plot_shadowed_data(discrete_data, colour, alpha_value):
    sampling_frequency = vf.get_param_value('sampling_frequency')
    time, dims = discrete_data.shape
    time_axis = np.arange(0,time)/float(sampling_frequency)
    plt.fill_between(time_axis,0,discrete_data[:,0],color=colour,alpha=alpha_value)

def plot_shadowed_data_time(time_axis, discrete_data, colour, alpha_value):
    plt.fill_between(time_axis,0,discrete_data[:,0],color=colour,alpha=alpha_value)


def plot_shadowed_data_non_zero(discrete_data, colour):
    sampling_frequency = vf.get_param_value('sampling_frequency')
    time, dims = discrete_data.shape
    time_axis = np.arange(0,time)/float(sampling_frequency)
    plt.fill_between(time_axis,discrete_data[:,0],discrete_data[:,0].max(),color=colour,alpha=0.5)

def plot_shadowed_data_new_class(discrete_data, colour, label_name):
    sampling_frequency = vf.get_param_value('sampling_frequency')
    time, dims = discrete_data.shape
    time_axis = np.arange(0,time)/float(sampling_frequency)
    plt.fill_between(time_axis,discrete_data[:,0],discrete_data[:,0].max(),color=colour,alpha=0.5, label=label_name)
    plt.plot(time_axis, discrete_data[:,0],color=colour,alpha=0.5, label=label_name)
    plt.legend()
def new_frequency_features(frequency_features, n_channels):
    time, dims = frequency_features.shape
    features_per_channel = int(dims/n_channels)
    new_feat_vect=np.zeros((time, n_channels))
    for chann_idx in range(n_channels):
        new_feat_vect[:, chann_idx] = frequency_features[:,int(chann_idx*features_per_channel):int((chann_idx+1)*features_per_channel)].mean(1)
    return new_feat_vect
    
def windows_derivatives(data, window_length):
    time, dims, trials = data.shape
    derivative_trials=np.zeros((time,dims,trials))
    for trial_idx in range(trials):
        diff_mat=np.zeros((time-1, dims))
        for time_idx in range(time-1):
            diff_mat[time_idx,:]=data[time_idx+1,:,trial_idx]-data[time_idx,:,trial_idx]
        derivative_trials[:,:,trial_idx] = diff_mat.mean(0)
    return derivative_trials
    
def create_pickle_dict(clf_list, title_list):
    '''Takes a list of classifiers and a list of titles
    then it bundles them in a dictionary, so its easier to manage
    using pickle afterwards.'''
    pickle_dict = dict(zip(title_list, clf_list))
    #Using zip on 2 lists we can populate a dictionary very fast
    return pickle_dict
    
    
def create_dataset(number_of_experiments, test_subject, experiment_months, file_list):
    '''
    Runs the necessary iterations and returns the feature datasets
    ----------------------------------------------------------------------
    input:
    ---------------------------------------------------------------------
    number_experiments : Is the number of experiments to run (constrained)
    test_subject: String with the subject name (as in the file)
    experiment_months : Months of the sessions
    All of these are constrained to the naming convention of the files
    ------------------------------------------------------------------------
    returns:
    ----------------------------------------------------------------------
    prehold_labels = labels for the prehold period
    hold_labels = labels for the hold period
    baseline_labels = labels for the baseline period
    cue_labels = labels that mark the cue moment
    data = array with all the pertinent data
    deriv_data = array with all the data for the derivatives
    '''
    experiment_days=np.arange(12,32)#days of the month for different runs
    #############################################################################
    #############################################################################
    #############Define Data#########################################################
    samp_frequency=vf.get_param_value('sampling_frequency')
    window_length_sec=vf.get_param_value('window_length_secs')
    t=np.arange(0,7,1.0/samp_frequency)
    features_file = file_list[0]
    deriv_features_file = file_list[1]
    ##################################################################################
    ###############################Start Iterations################################33
    ############################################################################
    for runs_month in experiment_months:
        #iterate for different days of the month
        for runs_days in experiment_days:
            #iterate over different runs in the day
            for run_index in range(1,number_of_experiments+1):
                #Check if the current set of experiments exists for the user
                #data_file='/home/leon/Data/Subjects/'+test_subject+'/Exp_'+str(runs_days)+'_'+str(runs_month)+'/Exp_'+str(run_index)+'/smoothbartlett.csv'
                data_file='/home/leon/Data/Subjects/'+test_subject+'/Exp_'+str(runs_days)+'_'+str(runs_month)+'/Exp_'+str(run_index)+features_file
                deriv_file='/home/leon/Data/Subjects/'+test_subject+'/Exp_'+str(runs_days)+'_'+str(runs_month)+'/Exp_'+str(run_index)+deriv_features_file
                if os.path.exists(data_file):
                    eeg_dataset=np.loadtxt(data_file,delimiter=',',dtype=float)[200:,:]
                    deriv_eeg_dataset=np.loadtxt(deriv_file,delimiter=',',dtype=float)[200:,:]
                    #time,dims=eeg_dataset.shape
                    #time_trunc=np.round(time*0.9)#This takes out the last trash samples
                    #eeg_dataset=eeg_dataset[500:time_trunc,:]
                    cue_labels=eeg_dataset[:,2]#Label information/ Cue
                    cursor=eeg_dataset[:,1]#The cursor information
                    frequency_features=eeg_dataset[:, 3:]#information of the rest of the features
                    eeg_data=frequency_features
                    deriv_eeg_data = deriv_eeg_dataset[:, 3:]
                    cue_idx=utils.getIndex(cue_labels,'up') ###Getting the indexes for the cue onset
                    prep_idx=utils.get_prep_Index(cue_labels,cue_idx)###Gets the index of the start of the preparation
                    baseline_start_idx=utils.getIndex(cue_labels,'down')
                    prehold_idx=utils.get_prehold_Index(cue_labels,prep_idx)
                    #################Set Datasets for the concat##########################
                    _baseline_labels_ = create_labels(baseline_start_idx, prehold_idx, eeg_data)
                    _prehold_labels_= create_labels(prehold_idx, prep_idx, eeg_data)
                    _cue_labels_=cue_labels[:,None]
                    _hold_labels_= create_labels(prep_idx, cue_idx, eeg_data)
                    ############################################################################
                    if run_index==1:
                        prehold_labels = _prehold_labels_
                        hold_labels = _hold_labels_#Creates a new set of labels where the hold is 1 and not hold is 0
                        cue_labels_data = _cue_labels_
                        baseline_labels = _baseline_labels_
                        data_ = eeg_data
                        deriv_data = deriv_eeg_data
                    else:
                        hold_labels=np.vstack((hold_labels, _hold_labels_))
                        data_=np.vstack((data_, eeg_data))
                        deriv_data = np.vstack((deriv_data, deriv_eeg_data))
                        baseline_labels=np.vstack((baseline_labels, _baseline_labels_))
                        cue_labels_data=np.vstack((cue_labels_data, _cue_labels_))
                        prehold_labels = np.vstack((prehold_labels, _prehold_labels_))
        return baseline_labels, prehold_labels, hold_labels, cue_labels_data, data_, deriv_data
        
        
