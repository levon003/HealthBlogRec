"""

Transformations are intended to support two operations:
 - Removal (dropping a column)
 - Transformation (dropping a column and replacing it with one or more new columns)
 
Supported transformation:
 - Categorical variable (n columns -> one column)
 - OneHotEncoding (one column -> n columns)

"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import sklearn.preprocessing

import cbrec.modeling.modelconfig


class Transformation:
    """
    TODO this is just a sketch at this point
    The idea of a transformation is that it takes 0 or more input data columns and produces 1 or more output columns.
    
    TODO is it reasonable to insist that a transformation return the number and names of the columns it produces before exposing it to data?
    TODO probably want some custom methods here to save and load the appropriate transformation
    """
    def __init__(self, feature_column_descriptor, drop_original=True):
        self.logger = logging.getLogger("cbrec.modeling.preprocessing.Transformation")
        self.feature_column_descriptor = feature_column_descriptor        
        self.drop_original = drop_original
        
        self.source_feature_names = []
        self.source_feature_indices = []
    
    def set_feature_columns(self, feature_manager):
        self.source_feature_names, self.source_feature_indices = self._get_features_from_column_descriptor(feature_column_descriptor)
        if len(self.source_feature_names) == 0:
            self.logger.warning("No features will be transformed for this transformation.")

    def transform(X):
        """
        :returns
            X_new - a matrix to be concatenated to X
            output_feature_names - a list of feature names, to be appended to feature_keys
        """
        raise ValueError("Not yet implemented! Implement in a subclass.")

        
class LogT:
    def __init__(self, feature_column_descriptor, eta=1):
        super().__init__(feature_column_descriptor)
        self.eta = eta
        
    def transform(X):
        X_new = np.log(X[:,self.source_feature_indices] + self.eta)
        output_feature_names = [fn + "_log" for fn in self.source_feature_names]
        return X_new, output_feature_names
        

class OneHotEncoding:
    def __init__(self, original_column_name: str, bins):
        self.original_column_name = original_column_name
        self.bins = bins
        if isinstance(bins, str):
            self.applyPreset()
    
    def applyPreset(self):
        if self.bins == "Duration":
            self.bins = [1/60, 1, 24, 24*7, 24*365]
        else:
            raise Exception("No preset defined for " + preset)


class FeatureManager:
    """
    Things this FeatureManager ought to be able to do:
     - Scale the data (and learn a scaling from the training data)
     - Transform a feature column in-place (such as applying a log transformation)
     - Remove a feature column and add 2 or more new columns based on that feature
         e.g. one-hot encoding a continuous feature that represents a duration
     - Remove a feature column
     
     What is an "activity" column? Any of a seed set of columns, plus any column derived from a seed column.
Same for "source" columns, etc.

Note: SOME transformations e.g. k-buckets, are learnable, and therefore need to be saved to disk.
    """
    def __init__(self, config: cbrec.modeling.modelconfig.ModelConfig):
        self.config = config
        self.logger = logging.getLogger("cbrec.modeling.preprocess.FeatureManager")
        
        self.scaler = None
        
        self.key_separator = "-"
        self.drop_indices = None
        
        self.__compute_feature_keys()
        #self.__compute_drop_indices()
    
    def __compute_feature_keys(self):
        """
        Compute the default feature keys expected in the input.
        """
        if self.config.data_version == '1.0.0':
            self.__compute_feature_keys_v1()
        else:
            raise ValueError(f"Data version {self.config.data_version} not recognized.")
            
    def __compute_feature_keys_v1(self):
        self.feature_keys = []
        
        for usp_type in ['source', 'candidate']:
            # 3 network features
            self.feature_keys.append(usp_type + self.key_separator + "indegree")
            self.feature_keys.append(usp_type + self.key_separator + "outdegree")
            self.feature_keys.append(usp_type + self.key_separator + "component_size")
            # 9 activity features
            for int_type in ['journal', 'amp', 'comment', 'guestbook']:
                self.feature_keys.append(usp_type + self.key_separator + int_type + "_count")
                self.feature_keys.append(usp_type + self.key_separator + int_type + "_time_to_most_recent")
            self.feature_keys.append(usp_type + self.key_separator + "time_to_first_update")
        # 3 shared network features
        self.feature_keys.append("shared" + self.key_separator + "are_weakly_connected")
        self.feature_keys.append("shared" + self.key_separator + "is_fof")
        self.feature_keys.append("shared" + self.key_separator + "is_reciprocal")
        
        # 768 text features
        for usp_type in ['source', 'candidate']:
            for i in range(768):  # number of text features
                self.feature_keys.append(usp_type + self.key_separator + "text" + str(i))
        assert len(self.feature_keys) == 1563
    
    def get_feature_index(self, feature_name):
        return self.feature_keys.index(feature_name)
    
    def get_feature_indices(self, usp_type, feature_descriptor, usp_type_inverse = False, feature_descriptor_inverse = False):
        """
        Allows one to retrieve one or more indices
        
        Examples:
        get_feature_indices(['source'], ['outdegree', 'indegree']) #source indegree and source outdegree
        get_feature_indices(['source','candidate'], ['outdegree', 'indegree']) #source indegree and source outdegree and candidate indegree and candidate outdegree
        get_feature_indices(['source'], '*')  # all source feature inds
        get_feature_indices('source', 'text')  # all source text features
        get_feature_indices('source', 'text_0')  # specific index
        get_feature_indices('source', 'text', feature_descriptor_inverse= True)  # all source non-text features
        
        get_feature_indices('*', 'cat')  # all categorical features
        
        Two special feature_descriptor keys refer to sets of features: 'network' and 'activity'
        
        :usp_type - valid values: source, candidate, shared, *
        :feature_descriptor - a string that will match with a feature that matches any part of a feature name
        """
        self.__compute_feature_keys()
        self.__compute_drop_indices()  # this function retrieves indices for the adjusted feature set; need to ensure we have dropped the appropriate columns from feature_keys
        # drop the associated feature keys (from right to left in the feature_keys list)
        for ind in self.drop_indices:
            self.feature_keys.pop(ind)            
        return self._get_feature_indices(usp_type, feature_descriptor, usp_type_inverse=usp_type_inverse, feature_descriptor_inverse=feature_descriptor_inverse)
        
    def _get_feature_indices(self, usp_type, feature_descriptor, 
                             usp_type_inverse=False, 
                             feature_descriptor_inverse=False,
                             include_feature_names=False):
        if feature_descriptor == 'network':
            return self._get_feature_indices(usp_type, 
                ['indegree', 'outdegree', 'component_size', 'are_weakly_connected', 'is_fof', 'is_reciprocal'],
                usp_type_inverse=usp_type_inverse, feature_descriptor_inverse=feature_descriptor_inverse,
            )
        if feature_descriptor == 'activity':
            return self._get_feature_indices(usp_type, 
                ['time_to_first_update'] + [int_type + "_count" for int_type in ['journal', 'amp', 'comment', 'guestbook']] + [int_type + "_time_to_most_recent" for int_type in ['journal', 'amp', 'comment', 'guestbook']],
                usp_type_inverse=usp_type_inverse, feature_descriptor_inverse=feature_descriptor_inverse,
            )
        
        if type(usp_type) is str:
            usp_type = [usp_type]
        if type(feature_descriptor) is str:
            feature_descriptor = [feature_descriptor]
            
        return_indices = []
        return_feature_names = []
        for i in range(len(self.feature_keys)):
            key_parts = self.feature_keys[i].split(self.key_separator)  # [0] -> usp_type, [1] -> feature_descriptor
            usp_match = False
            for usp in usp_type:
                if usp in key_parts[0]:
                    usp_match = True
                    break
                elif usp == "*":
                    usp_match = True
                    
            if usp_match == usp_type_inverse:  # check if we were supposed to find a usp match or not and stop early
                continue
            
            full_match = False
            for descriptor in feature_descriptor:
                if descriptor in key_parts[1]:
                    # TODO this is confusing; currently, we allow for a partial substring match anywhere in the feature key.
                    full_match = True
                    break
                elif descriptor == "*":
                    full_match = True
            
            if full_match != feature_descriptor_inverse:  # check if we were supposed to find a descriptor match or not and add to return
                return_indices.append(i)
                if include_feature_names:
                    return_feature_names.append(self.feature_keys[i])
        if include_feature_names:
            return return_indices, return_feature_names
        else:
            return return_indices
    
    def get_feature_name(self, index):
        """
        Deprecated. Probably don't use this function, it may produce unexpected behavior.
        """
        return self.feature_keys[index]
    
    def _get_features_from_column_descriptor(self, col):
        if type(col) == str:
            if col in self.feature_keys:
                ind = self.feature_keys.index(col)
                return [col,], [ind,]
            else:
                self.logger.warning(f"Failed to find non-existent column '{col}'.")
                raise ValueError(col)
        elif type(col) == list:
            usp_type, feature_descriptor = col
            usp_type_inverse, feature_descriptor_inverse = False, False
            if usp_type.startswith('~'):
                usp_type_inverse = True
                usp_type = usp_type[1:]
            if feature_descriptor.startswith('~'):
                feature_descriptor_inverse = True
                feature_descriptor = feature_descriptor[1:]
            feature_indices, feature_names = self._get_feature_indices(usp_type, feature_descriptor, usp_type_inverse=usp_type_inverse, feature_descriptor_inverse=feature_descriptor_inverse, include_feature_names=True)
            return feature_names, feature_indices
        else:
            raise ValueError(type(col))
    
    def __compute_drop_indices(self):
        """
        Computes self.drop_indices, which is a list of indices to drop in self.feature_keys
        """
        if self.drop_indices is None:
            inds_to_drop = []
            for col in self.config.preprocess_drop_columns:
                if type(col) == str:
                    if col in self.feature_keys:
                        remove_ind = self.feature_keys.index(col)
                        inds_to_drop.append(remove_ind)
                    else:
                        self.logger.warning(f"Tried to remove non-existent column '{col}'.")
                        raise ValueError(col)
                elif type(col) == list:
                    usp_type, feature_descriptor = col
                    usp_type_inverse, feature_descriptor_inverse = False, False
                    if usp_type.startswith('~'):
                        usp_type_inverse = True
                        usp_type = usp_type[1:]
                    if feature_descriptor.startswith('~'):
                        feature_descriptor_inverse = True
                        feature_descriptor = feature_descriptor[1:]
                    remove_inds = self._get_feature_indices(usp_type, feature_descriptor, usp_type_inverse=usp_type_inverse, feature_descriptor_inverse=feature_descriptor_inverse)
                    inds_to_drop.extend(remove_inds)
                else:
                    raise ValueError(f"Expected either a string name or a list to pass to get_feature_indices(). (got {type(col)})")
                
            inds_to_drop.sort(reverse=True)
            self.drop_indices = np.array(inds_to_drop).astype(int)
    
    def __preprocess(self, X):
        self.__compute_feature_keys()  # every time we preprocess we need to reset feature keys because we assume X is in its raw form
        # note that in the case where we want a 1->n column transformation where some of the new n columns are dropped, this is not possible in the current implementation
        # (since dropping happens before transformation)
        # if you want to drop some of the columns produced by a Transform, edit the Transform instead
        X = self.__remove_feature_columns(X)
        X = self.__encode_feature_columns(X)
        return X
    
    def __remove_feature_columns(self, X):
        self.__compute_drop_indices()
        
        # drop the associated feature keys (from right to left in the feature_keys list)
        for ind in self.drop_indices:
            self.feature_keys.pop(ind)
        
        # drop the columns from the data
        X = np.delete(X, self.drop_indices, 1)
            
        assert len(self.feature_keys) == X.shape[1], "Expected number of feature keys to match number of columns."
        return X
    
    def __encode_feature_columns(self, X):
        for hei, hot_encoding in enumerate(self.config.preprocess_encode_columns):
            # turn into object so we can apply bin presets
            if type(hot_encoding) is dict:
                hot_encoding = OneHotEncoding(hot_encoding["original_column_name"], hot_encoding["bins"])
            
            remove_index = self.feature_keys.index(hot_encoding.original_column_name)
            # save original values to calculate hot codes
            original_values = X[:,remove_index]
            X = np.delete(X, remove_index, 1)
            self.feature_keys.pop(remove_index)
            
            added_extra_columns = False
            if type(hot_encoding.bins) is int: # will only be set during first fit pass as long as the same config is used for further model tests
                new_values, bins = pd.qcut(original_values, hot_encoding.bins, labels=False, retbins=True)
                bins = bins[1:-1] # remove last bin because np.digitize uses the left bin edge by default
                self.logger.info(f"Binned {hot_encoding.original_column_name} into {hot_encoding.bins} bins {bins}")
                hot_encoding.bins = bins.tolist()
                self.config.preprocess_encode_columns[hei] = hot_encoding
            else:
                new_values = np.digitize(original_values, hot_encoding.bins, right=True)
                # need to do this so all hot encoded values are present, these are removed later (:-len(hot_encoding.bins))
                added_extra_columns = True
                new_values = np.append(new_values, range(len(hot_encoding.bins)))
            
            data_frame = pd.get_dummies(new_values, prefix=hot_encoding.original_column_name, prefix_sep="_cat_")
            new_hot_coded_values = data_frame.to_numpy()
                
            #splice new values/keys
            for i in range(new_hot_coded_values.shape[1]):
                if not added_extra_columns:
                    X = np.insert(X, remove_index + i, new_hot_coded_values[:,i], 1)
                else:
                    X = np.insert(X, remove_index + i, new_hot_coded_values[:-len(hot_encoding.bins),i], 1)
                
            self.feature_keys[remove_index:remove_index] = data_frame.columns.values
        return X
        
    def __fit(self, X):        
        # TODO set any learnable transformations from the data
        if self.config.preprocess_use_scaler:
            self.scaler = sklearn.preprocessing.StandardScaler()
            self.scaler.fit(X)
    
    def transform(self, X, preprocess=True):    
        if preprocess:
            X = self.__preprocess(X)
        # apply transformations
        if self.config.preprocess_use_scaler:
            X = self.scaler.transform(X)
        return X
    
    def fit_transform(self, X):
        # optional: remove this function if it's easier to implement just fit and transform
        X = self.__preprocess(X)
        self.__fit(X)
        return self.transform(X, preprocess=False)
    
    def save_learned_transformations(self):
        if self.config.preprocess_use_scaler:
            scaler_filepath = os.path.join(self.config.output_dir, self.config.output_basename + '_scaler.pkl')
            with open(scaler_filepath, 'wb') as outfile:
                pickle.dump(self.scaler, outfile, protocol=pickle.HIGHEST_PROTOCOL)      
    
    def load_learned_transformations(self):
        if self.config.preprocess_use_scaler:
            scaler_filepath = os.path.join(self.config.output_dir, self.config.output_basename + '_scaler.pkl')
            with open(scaler_filepath, 'rb') as infile:
                self.scaler = pickle.load(infile)
