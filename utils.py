###Utilities for the Kalman Filter
# New BSD License
#
# Copyright (c) 2007 - 2012 The scikit-learn developers.
# All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
'''
Utility functions taken from scikit-learn
'''
import var_file as vf
import inspect
import itertools
import pylab as plt
import numpy as np
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler


def array1d(X, dtype=None, order=None):
    """Returns at least 1-d array with data from X"""
    return np.asarray(np.atleast_1d(X), dtype=dtype, order=order)


def array2d(X, dtype=None, order=None):
    """Returns at least 2-d array with data from X"""
    return np.asarray(np.atleast_2d(X), dtype=dtype, order=order)


def log_multivariate_normal_density(X, means, covars, min_covar=1.e-7):
    """Log probability for full covariance matrices. """
    if hasattr(linalg, 'solve_triangular'):
        # only in scipy since 0.9
        solve_triangular = linalg.solve_triangular
    else:
        # slower, but works
        solve_triangular = linalg.solve
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(itertools.izip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probabily stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                      lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = solve_triangular(cv_chol, (X - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) + \
                                     n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


class Bunch(dict):
    """Container object for datasets: dictionary-like object that exposes its
    keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def get_params(obj):
    '''Get names and values of all parameters in `obj`'s __init__'''
    try:
        # get names of every variable in the argument
        args = inspect.getargspec(obj.__init__)[0]
        args.pop(0)   # remove "self"

        # get values for each of the above in the object
        argdict = dict([(arg, obj.__getattribute__(arg)) for arg in args])
        return argdict
    except:
        raise ValueError("object has no __init__ method")


def preprocess_arguments(argsets, converters):
    """convert and collect arguments in order of priority

    Parameters
    ----------
    argsets : [{argname: argval}]
        a list of argument sets, each with lower levels of priority
    converters : {argname: function}
        conversion functions for each argument

    Returns
    -------
    result : {argname: argval}
        processed arguments
    """
    result = {}
    for argset in argsets:
        for (argname, argval) in argset.iteritems():
            # check that this argument is necessary
            if not argname in converters:
                raise ValueError("Unrecognized argument: %s" % (argname,))

            # potentially use this argument
            if argname not in result and argval is not None:
                # convert to right type
                argval = converters[argname](argval)

                # save
                result[argname] = argval

    # check that all arguments are covered
    if not len(converters.keys()) == len(result.keys()):
        missing = set(converters.keys()) - set(result.keys())
        s = "The following arguments are missing: %s" % (list(missing),)
        raise ValueError(s)

    return result

def getIndex(labels,direction):
        #Function to get the changes from 0 to 1
        lab_1=labels[0]
        limit=len(labels)
        outidx=[]
        for idx in range(limit):
                lab=labels[idx]
                if lab_1==0 and lab==1 and direction=='up':
                        outidx.append(idx)
                if lab_1==1 and lab==0 and direction=='down':
                        outidx.append(idx)
                lab_1=labels[idx]
        return outidx    
def get_prep_Index(labels,cue_idx):
    #This gives the indexes corresponding to the startof the prep time
    sampling_frequency=vf.get_param_value('sampling_frequency')
    n_seconds=9.0
    time_step=1.0/sampling_frequency
    prep_time=n_seconds/time_step
    prep_idx=[]
    if cue_idx[0]-prep_time>0:
        for idx in range(len(cue_idx)):
            prep_idx.append(int(cue_idx[idx]-prep_time))
    else:
        for idx in range(len(cue_idx)-1):
            prep_idx.append(int(cue_idx[idx+1]-prep_time))
    return prep_idx

def get_prehold_Index(labels,prep_idx):
    #This gives the indexes corresponding to the startof the prehold time
    sampling_frequency=vf.get_param_value('sampling_frequency')
    n_seconds=1.0
    time_step=1.0/sampling_frequency
    prehold_time=n_seconds/time_step #prep time is hard coded for n_seconds
    prehold_idx=[]
    if prep_idx[0]-prehold_time>0:
        for idx in range(len(prep_idx)):
            prehold_idx.append(int(prep_idx[idx]-prehold_time))
    else:
        for idx in range(len(prep_idx)-1):
            prehold_idx.append(int(prep_idx[idx+1]-prehold_time))
    return prehold_idx

def get_onset_Index(labels,cue_idx,time_limit):
    #This gives the indexes corresponding to the end of the onset period
    sampling_frequency=vf.get_param_value('sampling_frequency')
    n_seconds=4.0 #number of seconds
    time_step=1.0/sampling_frequency
    prep_time=n_seconds/time_step
    prep_idx=[]
    if cue_idx[-1]+prep_time>time_limit:
        for idx in range(len(cue_idx)):
            prep_idx.append(int(cue_idx[idx]+prep_time))
    else:
        for idx in range(len(cue_idx)-1):
            prep_idx.append(int(cue_idx[idx]+prep_time))
    return prep_idx
def get_baseline_data(start_indexes,end_indexes,data):
    '''This functions concatenates all the baslines'''
    '''Returns a large basline dataset'''
    while start_indexes[0]>end_indexes[0] or start_indexes[0]>end_indexes[0]:
         #This is to avoid incomplete datasets
        if start_indexes[0]>end_indexes[0]:
            del end_indexes[0]
        else:
            del start_indexes[0]
    while len(start_indexes)!=len(end_indexes):
        if len(start_indexes)<len(end_indexes):
            del end_indexes[-1]
        if len(start_indexes)>len(end_indexes):
            del start_indexes[-1]
    #setting up the data structure
    #set the case for data that is 1D
    if len(data.shape)<2:
        data=data[:,None]
    time,dims=data.shape
    trials=len(start_indexes)
    output_data=output_data=np.zeros((1,dims))
    join_indexes=[]
    for idx in range(trials):
        output_data=np.vstack((output_data,data[start_indexes[idx]:end_indexes[idx],:]))
        join_indexes.append(output_data.shape[0]-1)
    #join_indexes keeps track of where the data was pasted for future reference
    #Eliminate the dummy zeros from the beggining
    output_data=np.delete(output_data,0,0)
    return output_data, join_indexes

def get_data(start_indexes,end_indexes,data):
#This gets the segments for fixed length training samples
    if start_indexes[0]>end_indexes[0]:
        #This is to avoid incomplete datasets
        del start_indexes[0]
    while len(start_indexes)!=len(end_indexes):
        if len(start_indexes)<len(end_indexes):
            del end_indexes[-1]
        if len(start_indexes)>len(end_indexes):
            del start_indexes[-1]
    if len(set(np.array(end_indexes)-np.array(start_indexes)))>1:
        print "There is an error in your set"
        return None
    #setting up the data structure
    #set the case for data that is 1D
    if len(data.shape)<2:
        data=data[:,None]
    time,dims=data.shape
    trials=len(start_indexes)
    trial_time=len(range(start_indexes[0],end_indexes[0]))
    output_data=np.zeros((trial_time,dims,trials))
    for idx in range(trials):
        output_data[:,:,idx]=data[start_indexes[idx]:end_indexes[idx],:]
    
    return output_data
       

def homogenize_indexes(cIdx,nIdx):
    #This scripts makes sure both indexes have the same length
    length_pos_index=len(cIdx)
    length_neg_index=len(nIdx)
    if length_neg_index>length_pos_index:
        while(length_neg_index>length_pos_index):
            nIdx=nIdx[:-1] #Take out the last element
            length_neg_index-=1 #Reduce counts
    else:
        while(length_neg_index<length_pos_index):
            cIdx=cIdx[:-1] #Take out the last element
            length_pos_index-=1 #Reduce counts
    return cIdx,nIdx
        
def getSegment(cIdx,nIdx,length,pointer,labels):
        maxNW=0
        maxPW=0
        outidx=[]
        if pointer=='up':
                counts=len(cIdx)
                for idx in range(counts-1):
                        #print idx
                        #print range(0,nIdx[idx]-1)
                        if idx==0:
                                outidx.append(range(0,nIdx[idx]))

                        else:
                                outidx.append(range(nIdx[idx-1],nIdx[idx]))

        if pointer=='down':
                counts=len(nIdx)
                for idx in range(counts-1):
                        
                        #print idx
                        #print range(0,nIdx[idx]-1)
                        if idx==0:
                                outidx.append(range(cIdx[0],cIdx[idx+1]))
                        else:
                                outidx.append(range(cIdx[idx],cIdx[idx+1]))
                     
        return outidx
def getSpecArray(sIdx,data):
        #This function accomodates in a 3D array the segments for each frequency
        row,col=np.shape(sIdx)
        dumm,depth=np.shape(data)
        outidx=np.empty((depth,row,col))
        for lay in range(depth):
                for idx in range(row):
                        print sIdx[idx][-1],dumm
                        if sIdx[idx][-1]<dumm:
                                outidx[lay,idx,:]=data[sIdx[idx],lay]
                        else:
                                outidx[lay,idx,:]=np.zeros(col)
        return outidx

def CreateTrainingArray(data,indexes,lab):
        Midx=lab[indexes[0]][0]
        
        if len(np.shape(data))==1:
                data=data[:,None]
        trow,tcol=np.shape(data)
        length_half_mask_pos=max([np.sum((lab[idx]==1-Midx)) for idx in indexes])
        length_half_mask_neg=max([np.sum((lab[idx]==Midx)) for idx in indexes])
        MaskDataPos=np.zeros((length_half_mask_pos,tcol,len(indexes)))
        MaskDataNeg=np.zeros((length_half_mask_neg,tcol,len(indexes)))
        Ntrain=0
#        TrainOut=np.zeros((1,tcol))
        for idx in indexes:
                dummy=lab[idx]
                dummydata=data[idx]
                x=dummydata[np.nonzero(dummy==1-Midx)]
                y=dummydata[np.nonzero(dummy==Midx)]
                if len(x)<=length_half_mask_pos:
                        
                        limit=x[-1,:][None,:]
                        limit=np.tile(limit,(length_half_mask_pos-len(x),1))
                        x=np.vstack((x,limit))
                if len(y)<=length_half_mask_neg:
                        limit=y[0,:][None,:]
                        limit=np.tile(limit,(length_half_mask_neg-len(y),1))
                        y=np.vstack((limit,y))
                MaskDataPos[:,:,Ntrain]=x
                MaskDataNeg[:,:,Ntrain]=y
                Ntrain+=1

        return np.concatenate((MaskDataNeg,MaskDataPos))
def depurate_segments(p_idx,n_idx,data_labels):
    new_p_idx=[]
    new_n_idx=[]
    for idx in p_idx:
        if np.any(data_labels[idx]==0) and np.any(data_labels[idx]==0): #check if it has both zeros and ones
            new_p_idx.append(idx)
    for idx in n_idx:
        if np.any(data_labels[idx]==0) and np.any(data_labels[idx]==0): #check if it has both zeros and ones
            new_n_idx.append(idx)
    return new_p_idx,new_n_idx
            
def CreateAverages(data,indexes,lab):
        #creating the positive mask
        arrNeg=np.ma.empty((1,355,15))
        arrPos=np.ma.empty((1,355,15))
        arrNeg.mask=True
        arrPos.mask=True
        time,channels=np.shape(data)
        masterAr=np.zeros((channels,355*2))
        for chan in range(channels):
                for i in range(13):
                        dummy=lab[indexes[i]]
                        dummydata=data[indexes[i],chan][:,None]
                        x=dummydata[np.nonzero(dummy==1),0]#The dummy takes care of the indexes
                        #We need to turn the x, due to the alignment
                        x=x[:,::-1]
                        
                        arrPos[:x.shape[0],:x.shape[1],i]=x
                        x=dummydata[np.nonzero(dummy==0),0]#This magic instruction takes care of taking only the positive side
                        arrNeg[:x.shape[0],:x.shape[1],i]=x
                        Half1=arrPos.mean(axis=2)
                        #We need to do the reinversion
                        Half1=Half1[:,::-1]
                        Half2=arrNeg.mean(axis=2)
                        out=np.hstack((Half2,Half1))
               
                masterAr[chan,:]=out                
        
        plt.figure()
        plt.subplot(111)
        timeZ=np.arange(-18,18,2)
        plt.plot(masterAr)
        #plt.imshow(masterAr,aspect='auto',interpolation='none',origin='lower')
        #plt.colorbar()
        #plt.(masterAr.T[:,:])
        
        
        #title='Heat Map for the Averaged FA Components (1-10)'
        title='Heat Map for the Averaged Components (0->1)'
        plt.title(title)
        plt.yticks(np.arange(0,chan+1),np.arange(1,chan+2))
        #plt.xticks(np.arange(0,355*2,355*2/18),timeZ)
        plt.xlabel('Time[s]')
        plt.ylabel('Components')
        #plt.ylabel('Components')
        #plt.axvline(x=353,linewidth=2, color='black')
        
        plt.savefig('AveragedFreq110FA',dpi=300)
        #plt.show()
        
        return 0
def CreateAveragesNeo(data,indexes,pind,lab,posit,flag,labl,fill):
        #creating the positive mask
        arrNeg=np.ma.empty((1,355,15))
        arrPos=np.ma.empty((1,355,15))
        arrNeg.mask=True
        arrPos.mask=True
        time,channels=np.shape(data)
        masterAr=np.zeros((channels,355*2))
        for chan in range(channels):
                for i in range(15):
                        dummy=lab[indexes[i]]
                        dummydata=data[indexes[i],chan][:,None]
                        x=dummydata[np.nonzero(dummy==1),0]#The dummy takes care of the indexes
                        #We need to turn the x, due to the alignment
                        x=x[:,::-1]
                        
                        arrPos[:x.shape[0],:x.shape[1],i]=x
                        x=dummydata[np.nonzero(dummy==0),0]#This magic instruction takes care of taking only the positive side
                        arrNeg[:x.shape[0],:x.shape[1],i]=x
                        Half1=arrPos.mean(axis=2)
                        #We need to do the reinversion
                        Half1=Half1[:,::-1]
                        Half2=arrNeg.mean(axis=2)
                        out=np.hstack((Half2,Half1))
               
                masterAr[chan,:]=out                
        
        
        
        timeZ=np.arange(-18,18,2)
        
        print timeZ
        print np.arange(0,355*2,355*2/18.)
        if flag:
                for i in range(channels):
                        plt.plot(masterAr.T[:,i], label='Comp '+str(i+1))
                
        else:
                plt.plot(masterAr.T[:,:],label=labl)
                if fill:
                        plt.fill_between(np.arange(0,710,1),masterAr.T[:,0],color='0.8')
        plt.legend(loc=3,prop={'size':9})
        title='Averaged PLS Regression ((0-40[Hz])Frequencies, Smoothing, Training=1/4)'
        plt.title(title)
        plt.xlabel('Time(sec)')
        plt.xticks(np.arange(0,355*2,355*2/18),timeZ)
        plt.ylabel('Cursor Position/Label')
        plt.axvline(x=352,linewidth=2, color='black')
        xinf,xsup,yinf,ysup= plt.axis()
        #plt.text(330,yinf+0.08*yinf,'0.0')
        #plt.ylabel('Components Magnitude')
        
        #plt.show()
        
        return 0

def homogenize_data(data1,data2):
#If 2 arrays have different sizes, it stadardizes them to the smallest
        time_1,dims_1,tran_1=data1.shape
        time_2,dims_2,tran_2=data2.shape
        if (time_1==time_2) and (tran_1==tran_2):
                return data1,data2
        else:
                if tran_1>tran_2:
                        while tran_1>tran_2:
                                data1=data1[:,:,:-1]
                                tran_1-=1
                if tran_2>tran_1:
                        while tran_2>tran_1:
                                data2=data2[:,:,:-1]
                                tran_2-=1
                return data1,data2
def change_mix_to_states(mixture_matrix):
        #Given a vector of probabilities outputs a vector of states
        mixture_matrix_temp=np.copy(mixture_matrix)
        np.putmask(mixture_matrix_temp,mixture_matrix_temp>=0.5,1)
        np.putmask(mixture_matrix_temp,mixture_matrix_temp<=0.5,0)
        return mixture_matrix_temp

def plot_lds_samples(lds,data_lds, x_label, y_label, title):
    ''' Plots samples from the respective filters for each cluster'''
    samp_frequency = vf.get_param_value('sampling_frequency')
    n_samples=vf.get_param_value('LDS_Samples')
    time, dims, trials = np.shape(data_lds)
    time_axis = np.arange(0,time)/float(samp_frequency)
    for i in range(n_samples):
        plot_sample=lds.sample(int(time))[1]
        plt.plot(time_axis, plot_sample[:,0], 'black')
        plt.plot(time_axis, plot_sample[:,1], 'blue')
        plt.plot(time_axis, plot_sample[:,2], 'red')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.grid()

def subplot_lds_samples(lds_list, data_list):
    title=['Baseline Samples', 'Hold Samples', 'Cue Onset Samples', 'Cue Hold Samples']
    h = np.ceil(np.sqrt(len(lds_list)))
    figu = plt.figure()
    for filter_idx in range(len(lds_list)):
        ax = figu.add_subplot(h,h,filter_idx+1)
        plot_lds_samples(lds_list[filter_idx], data_list[filter_idx], 'Time', 'Magnitude', title[filter_idx])

def plot_cdf(mix_probs,threshold,flag):

        time_mix,dim_mix,samples_mix=mix_probs.shape
        firing_vect=np.zeros((time_mix,1,samples_mix))
        for sdx in range(samples_mix):
                if np.nonzero(mix_probs[:,flag,sdx]>threshold)[0].shape[0]!=0:
                        idx=np.nonzero(mix_probs[:,flag,sdx]>threshold)[0][0]
                        print sdx,idx
                        firing_vect[idx:,0,sdx]=1
        return firing_vect

def plot_IMM_mixing_matrix(mixing_matrix, label, *args):
        '''Plots the mixing matrix of the IMM, the last element of args is the cue'''
        samp_frequency = vf.get_param_value('sampling_frequency')
        time, dims, = np.shape(mixing_matrix)
        time_axis = np.arange(0,time)/float(samp_frequency)
        state_list=[]
        for state in args:
            state_list.append(state)
        plt.figure()
        plt.plot(time_axis, mixing_matrix)
        plt.plot(time_axis, label)
        plt.xlabel('Time')
        plt.ylabel('Probability')
        plt.title('States Probabilities')
        plt.legend(state_list)
        plt.grid()
def segment_training_data(data, labels, test_size=0.5):
    '''This takes the data and segments it in joint sets of data'''
    if test_size==0:
        data_train=data
        data_test=data
        labels_train=labels
        labels_test=labels
    else:
        time, dims=data.shape
        limit_idx=round((1-test_size)*time)
        data_train=data[0:limit_idx,:]
        data_test=data[limit_idx:,:]
        labels_train=labels[0:limit_idx,:]
        labels_test=labels[limit_idx:,:]
    return data_train, data_test, labels_train, labels_test

def return_plots(mix_data,data_trend):
        if data_trend=='up':
                idx=0
        if data_trend=='down':
                idx=1
        sca=MinMaxScaler()# class to normalize data to [0,1]
        mixture_histogram_data=mix_data[:,:,:].sum(2)
        time_histogram=np.linspace(-18,18,len(mixture_histogram_data))
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        # Set common labels
        ax.set_xlabel('Time[s]')
        ax.set_ylabel('Probabilities')
        ax.set_title('Probability for Up State (Down to Up)')

        plt.subplots_adjust(hspace=0.100)
        number_of_subplots=mix_data.shape[2]
        for i,v in enumerate(xrange(number_of_subplots)):
            v = v+1
            ax1 = fig.add_subplot(number_of_subplots,1,v)
            ax1.plot(time_histogram[315:],mix_data[315:,idx,i])
            plt.axvline(x=0,linewidth=2, color='black')
            ticks=np.floor(np.arange(time_histogram[315:].min(),time_histogram[315:].max(),2))
            plt.xticks(ticks)
            plt.grid()
        
        plt.figure(2, figsize=(16, 9))
        for cdf_thres in np.arange(0.5,1,0.1):
                label_plot='Treshold = '+str(cdf_thres)
                firings=plot_cdf(mix_data[315:,:,:],cdf_thres,idx)
                firing_plot=sca.fit_transform(firings.sum(2))
                plt.plot(time_histogram[315:],firing_plot,label=label_plot)
        ticks=np.floor(np.arange(time_histogram[315:].min(),time_histogram[315:].max(),2))
        plt.xticks(ticks)
        plt.grid()
        plt.axvline(x=0,linewidth=2, color='black')
        plt.legend(loc='lower right')
        plt.ylabel('Normalized Counts')
        plt.xlabel('Time[s]')
        plt.title('CDF for testing trials')
        
def create_channels_mask(Num_Channels,dimensions_input):
    features_per_channel=dimensions_input/Num_Channels
    channel_mask=np.zeros((Num_Channels,dimensions_input))
    for cidx in range(Num_Channels):
        channel_mask[cidx,cidx*features_per_channel:cidx*features_per_channel+features_per_channel]=1
    return channel_mask


def add_matlab_element(matlab_dict, **kwargs):
    for key, value in kwargs.iteritems():
        matlab_dict[key]=value
    return matlab_dict
    
def getWindows(data ,left_length, right_length, data_idx):
    ''' This function uses the data, the left and right bounds and the index'''
    ''' The bounds are given in seconds '''
    ''' The indexes are calculated before and indicate the crossings '''
    if len(data.shape)<2:
        data = data[:, None]
    time, dims = data.shape
    sampling_frequency = vf.get_param_value('sampling_frequency')
    left_bound = np.round(sampling_frequency*left_length)
    right_bound = np.round(sampling_frequency*right_length)
    row_length = left_bound + right_bound
    trial_length = len(data_idx)
    output_array = np.zeros((row_length, dims, trial_length))
    time_axis = np.linspace(-left_length, right_length, row_length)
    for d_idx in range(len(data_idx)):
        left_idx = -left_bound+data_idx[d_idx]
        right_idx = data_idx[d_idx] + right_bound
        if left_idx < 0:
            pass
        if right_idx > time:
            pass
        else:
            output_array[:, :, d_idx] = data[left_idx:right_idx, :]
    return output_array, time_axis
    
def detect_first_pass(binary_data):
    '''Detects the first time the bit is up'''
    '''Only accepts binary data'''
    ''' The data has to be in windowed trials'''
    if len(np.unique(binary_data).shape) > 2:
        print 'This is not a binary array'
    time, dims, trials = np.shape(binary_data)
    switch_array = np.zeros((time, dims, trials))
    for trial_idx in range(trials):
        if binary_data[0,0,trial_idx] == 1:
            switch_array[0, 0, trial_idx] = 1
        else:
            switch_index = np.nonzero(np.diff(binary_data[:,0,trial_idx]))[0] + 1
            if switch_index.size == 0 or switch_index[0] > time:
                pass
            else:
                switch_array[switch_index[0], 0,  trial_idx] = 1
        #gets the switiching indexes -1
    return switch_array

    
    
    
    
    
    
    
    
    
    
    
    
