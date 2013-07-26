#This library is an example of how to visualize the EEG topography using python
#Similar examples should be easy to do in matlab
import eeg_topo_utils as tutils
import numpy as np
import pylab as plt
import pickle
pickle_filename = 'pickle_class_8_12.p'
clf_dict = pickle.load(open(pickle_filename, "rb"))
clf = clf_dict['Derivatives']
clf_features = 'Single Channel'
tutils.plot_top_clf(clf, clf_features, 'Power')
plt.figure()
random_feat = np.random.rand(1,60)
tutils.plot_top_array(random_feat, 'Single Channel', 'Power')
