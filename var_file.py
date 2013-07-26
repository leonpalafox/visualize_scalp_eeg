#####Master file woth the value for eveyr variable in the different scripts
import genUtils as gu #The python swith is defined here

####Variables related with the filters

def get_param_value(var_name):
    for case in gu.switch(var_name):
                if case ('window_length_secs'):
                        value=0.1
                        break
                if case('number_of_filters'):
                        value = 2
                        break
                if case('EM_Iterations'):
                        value = 10
                        break
                if case('EM_latent_variables'):
                        value=6
                        break
                if case('PLS_Components'):
                        value = 30
                        break
                if case('LDS_Samples'):
                        value = 20
                        break
                if case('Training_percentage'):
                        value=0.6   
                        break
                if case('TrainSamp'):
                        value=2
                        break
                if case('number_channels'):
                        value=64
                        break
                if case('IMM_Training_percentage'):
                        value=1
                        break
                if case('pca_components'):
                        value=300
                        break
                if case('ica_components'):
                        value=50
                        break
                if case('number_of_clusters'):
                        value=4
                        break
                if case('sampling_frequency'):
                        value=256/13.0
                        break
                if case():
                        print 'No value'
    return value

