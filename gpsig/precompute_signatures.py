import hashlib
import json
import os
import pickle
import numpy as np
import iisignature
from scipy.optimize import brentq as brentq

# Adapted from the drone_identification repository in the DataSig GitHub 

# This is how to define a decorator function in Python.
# See https://wiki.python.org/moin/PythonDecorators.
# We use this function to cache the results of calls to
# compute_signature().
def cache_result(function_to_cache, cache_directory='cached_signatures'):
    """
    Cache the result of calling function_to_cache().
    """
    if not os.path.isdir(cache_directory):
        os.mkdir(cache_directory)

    def _read_result(parameter_hash):
        cache_file = os.path.join(cache_directory, str(parameter_hash))
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                return pickle.load(file)
        return None

    def _write_result(parameter_hash, result):
        cache_file = os.path.join(cache_directory, str(parameter_hash))
        with open(cache_file, 'wb') as file:
            pickle.dump(result, file)

    def wrapper_function(*args, **kwargs):
        # Aggregate all parameters passed to function_to_cache.
        # Note that args[0] will contain an instance of ExpectedSignatureCalculator(), which
        # we subsequently convert to a string. In this way, we cache results, while
        # considering values of constants and the truncation level.
        parameters = args + tuple(kwargs[k] for k in sorted(kwargs.keys()))
        # Map aggregated parameters to an MD5 hash.
        parameter_hash = hashlib.md5(json.dumps(parameters, sort_keys=True,
                                                default=str).encode('utf-8')).hexdigest()

        result = _read_result(parameter_hash)
        if result is None:
            # Call function_to_cache if no cached result is available
            result = function_to_cache(*args, **kwargs)
            _write_result(parameter_hash, result)

        return result

    return wrapper_function

class SignatureCalculator():
    """
    Class for computing the path signatures.
    """

    def __init__(self, dataset, dataset_type, truncation_level=5, add_time=False,
                 lead_lag_transformation=False, Harald_rescaling=-1):
        """
        Parameters
        ----------
        dataset : string
            Name of the dataset
        dataset_type: string
            Whether it is the train, validation or the test set
        truncation_level : int
            Path signature trucation level.
        add_time: bool
            Whether add time was applied
        lead_lag_transformation : bool
            Whether lead-lag was applied.
        """

        assert dataset_type in ['train', 'val', 'test'], "dataset_type should be 'train', 'val' or 'test'"

        self.dataset = dataset
        self.dataset_type = dataset_type
        self.truncation_level = truncation_level
        self.add_time = add_time
        self.lead_lag_transformation = lead_lag_transformation
        self.Harald_rescaling = Harald_rescaling

    @cache_result
    def compute_signature(self, data):
        if self.Harald_rescaling!=-1:
            data = scale_paths(X=data,M=self.Harald_rescaling,a=1,level_sig=self.truncation_level)
            return [iisignature.sig(data, self.truncation_level), data]
        return iisignature.sig(data, self.truncation_level)

    def __str__(self):
        """
        Convert object to string. Used in conjunction with cache_result().
        """
        return str((self.dataset,
                    self.dataset_type,
                    self.truncation_level,
                    self.add_time,
                    self.lead_lag_transformation,
                    self.Harald_rescaling))


''' MAIN FUNCTIONS USED FOR PATH RE-SCALING '''

def scale_paths(X,M,a,level_sig):
    '''
        This function computes a single scaling factor \theta_x (path-dependent) per path and rescales each path x 
        , i.e. return x_new: t -> \theta_x*x_t, such that ||S_{<level_sig}(x_new)|| < M(1+1/a).
        
        Inputs:
            - X: an array (N,L,D) representing N paths where L is the length and D is the state-space dimension
            - (int) level_sig: the truncation level to compute the norm of S_{<level_sig}(x)
            - (int) M: the first parameter used to define the maximum norm allowed 
            - (float) a: the second parameter used to define the maximum norm allowed 
        Outputs:
            - X_new: the rescaled paths 
    '''
    N = X.shape[0]
    D = X.shape[2]

    maxi = M*(1.+1./a)

    Sig = iisignature.sig(X, level_sig) # computes the signatures of the paths in X

    norms_levels = get_norms_level_sig(Sig, D, level_sig) # (N,n) each row contains the (squared) norm of each tensor in S(x_i) truncated at level level_sig=n
    
    norms = np.sum(norms_levels,axis=1) # (N,1) gets the (squared) norms of S(x) truncated at level level_sig 

    psi = [psi_tilde(norms[i],M,a) for i in range(N)]  # computes an intermediary number which is used to find the scale

    thetas = [brentq(poly,0, 10000, args=(psi[i], norms_levels[i],level_sig)) for i in range(N)]# computes the scale 

    return np.array(thetas)[:,None,None]*X

''' UTILS FUNCTIONS ''' 

def get_norms_level_sig(Sig, d, level_sig):
    ''' 
        This function computes the norm of each tensor in the truncated signature in input
        INPUTS:
            - (array) Sig: flat signatures (N,M)
            - (int) d: the original state-space dimension of the paths
            - (int) level_sig: the truncation level of the signatures
        OUTPUT:
            - (list) norms: a list containing the norms of each level of the truncated signature in input
    '''
    norms = np.ones((Sig.shape[0],level_sig+1))

    for k in range(1,level_sig+1):

        start = int(((1 - d ** k) / (1 - d)) - 1)
        end = int((1 - d ** (k + 1)) / (1 - d) - 1)

        norms[:,k] = np.sum(Sig[:,start:end]**2,axis=1) # (1,N)

    return norms # (N,n)

def psi_tilde(x, M, a):
    '''
        psi(\sqrt{x}) =  x if x<=M; M+M^{1+a}(M^{-a}-x^{-a})/a otherwise
        this returns psi(||S_M(x)||) with ||S_M(x)||^2 given in input
    '''
    if x <= M:
        return x
    else:
        return M + pow(M, 1. + a) * (pow(M, -a) - pow(x, -a)) / a


def poly(x,psi,coef,level_sig):
    return np.sum([coef[i]*x**(2*i) for i in range(level_sig+1)])-psi



