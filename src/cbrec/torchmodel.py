
import logging
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pickle
import sklearn.preprocessing

import torch
import torchvision
import torchvision.transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

from . import featuredb
from . import reccontext
from . import evaluation
from . import data
from .text import embeddingdb
from .text import journalid


class LinearNet(nn.Module):
    """
    Simple neural net with 2 hidden layers.
    """
    def __init__(self, n_input, n_hidden, dropout_p=0.2):
        super(LinearNet, self).__init__()
        # note: 768 is the size of the roBERTa outputs
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1, bias=False)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # note: not using F.sigmoid here, as the loss used includes the Sigmoid transformation
        return x
    
    
class ConcatNet(nn.Module):
    """
    Takes two features sets (a, b) (that are nominally text features & non-text features) and concatenates them after a pass through a hidden layer
    """
    def __init__(self, n_input_a, n_input_b, n_hidden, dropout_p=0.2):
        super(ConcatNet, self).__init__()
        
        self.n_input_a = n_input_a
        self.n_input_b = n_input_b
        
        self.fc1_a = nn.Linear(n_input_a, n_hidden)
        self.fc1_b = nn.Linear(n_input_b, n_hidden)
        
        self.fc2 = nn.Linear(n_hidden*2, n_hidden)
        self.fc3 = nn.Linear(n_hidden, 1, bias=False)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        
    def forward(self, x):
        x_a = x[:self.n_input_a]
        x_b = x[self.n_input_a:]
        x_a = F.relu(self.fc1_a(x_a))
        x_b = F.relu(self.fc1_b(x_b))
        x = torch.cat([x_a, x_b], 0)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # note: not using F.sigmoid here, as the loss used includes the Sigmoid transformation
        return x
    

def train_pytorch_model(X_train, y_train, X_test, y_test):
    logger = logging.getLogger("cbrec.torchmodel.train_pytorch_model")
    
    n_train = len(y_train)
    n_test = len(y_test)
    
    verbose = True
    n_hidden = 100
    n_epochs = 800
    lr_init = 0.01
    max_lr = 0.0155 #0.038
    dropout_p = 0.1
    minibatch_size = len(y_train)
    minibatch_size = min(n_train, minibatch_size)  # if minibatch_size is larger than n_train, force it to n_train
    n_minibatches = int(np.ceil(n_train / minibatch_size))
    validation_rate = 0.1 # (vr) we will compute loss and accuracy against the validation set on vr of the epochs
    
    n_input = X_train.shape[1]
    # note: input dim is 27 for non-text features + 768 for text features
    net = LinearNet(n_input, n_hidden, dropout_p)
    
    #optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr_init)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=n_minibatches,
        epochs=n_epochs,
    )
    
    criterion = nn.BCEWithLogitsLoss()  # pointwise loss function
    
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test)
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    y_train_tensor = y_train_tensor.view(-1, 1)  # make labels 2-dimensional
    y_train_tensor = y_train_tensor.type_as(X_train_tensor)
    if verbose:
        logger.info(f"Input tensor sizes: {X_train_tensor.size()}, {y_train_tensor.size()}")
        logger.info(f"Validating model every {int(1/validation_rate)} epochs for {n_epochs} epochs.")
    
    # _metrics[0] -> Epoch, metrics[1] -> loss, _metrics[2] -> accuracy
    test_metrics = np.zeros((3,int(n_epochs*validation_rate+1))) #+1 to ensure space for final epoch metric
    train_metrics = np.zeros((3,n_epochs))
    
    for epoch in range(n_epochs):
        s = datetime.now()
        optimizer.zero_grad()
        
        # shuffle the training data
        # I am not sure if this matters at all
        epoch_order = torch.randperm(n_train)
        
        mb_metrics = []  # store the minibatch_metrics, then average after
        for minibatch in range(n_minibatches):
            minibatch_start = minibatch * minibatch_size
            minibatch_end = min(minibatch_start + minibatch_size, n_train)
            if verbose and epoch == 0:
                logger.info(f"    Minibatch for inds in {minibatch_start} - {minibatch_end}.")
            minibatch_inds = epoch_order[minibatch_start:minibatch_end]
            
            inputs = X_train_tensor[minibatch_inds]
            train_labels = y_train_tensor[minibatch_inds]

            net.train()
            train_outputs = net(inputs)
            train_loss = criterion(train_outputs, train_labels)
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            # compute accuracy
            y_train_pred = torch.sigmoid(train_outputs.detach()).view((-1,)).numpy()
            y_train_pred = (y_train_pred >= 0.5).astype(int)  # binarize predictions with a 0.5 decision boundary
            y_train_minibatch = y_train[minibatch_inds.numpy()]
            train_acc = np.sum(y_train_pred == y_train_minibatch) / len(y_train_minibatch)

            mb_metrics.append((train_loss.item(), train_acc))
        train_loss, train_acc = np.mean(np.array(mb_metrics), axis=0)
        train_metrics[0,epoch] = epoch
        train_metrics[1,epoch] = train_loss
        train_metrics[2,epoch] = train_acc
            
        should_stop_early = train_loss < 0.001
        if verbose and (epoch < 3 or epoch == n_epochs - 1 or epoch % 10 == 0 or should_stop_early):
            logger.info(f"{epoch:>3} ({datetime.now() - s}): train loss={train_loss:.4f} train accuracy={train_acc*100:.2f}% LR={optimizer.param_groups[0]['lr']:.2E}")
        if should_stop_early:
            break
            
        if epoch % (1/validation_rate) == 0:
            net.eval()
            with torch.no_grad():
                test_outputs = net(X_test_tensor)
                test_loss = criterion(test_outputs.detach(), y_test_tensor.unsqueeze(1).float())
                y_test_pred = torch.sigmoid(test_outputs.detach()).view((-1,)).numpy()
                y_test_pred = (y_test_pred >= 0.5).astype(int)
                test_acc = np.sum(y_test_pred == y_test) / len(y_test)
            logger.info(f"    {epoch:>3}: test loss={test_loss:.4f} test accuracy={test_acc*100:.2f}%")
            metric_ind = int(epoch*validation_rate)
            if test_loss <= np.min(test_metrics[1,:]):
                # this is the lowest loss we've reached
                logger.info(f"    Best validation lost achieved so far.")
            test_metrics[0,metric_ind] = epoch
            test_metrics[1,metric_ind] = test_loss
            test_metrics[2,metric_ind] = test_acc
        
    # this is a hack, but we store training results info back through the learner_config dictionary
    final_train_loss = train_loss
    final_epoch_count = epoch + 1
    if verbose:
        logger.info(f"Completed {final_epoch_count} epochs with a final train loss of {final_train_loss:.4f}.")
        
    net.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test)
        outputs = net(X_test_tensor)
        test_loss = criterion(test_outputs.detach(), y_test_tensor.unsqueeze(1).float())
        y_test_pred = torch.sigmoid(outputs.detach()).view((-1,)).numpy()
        y_test_pred = (y_test_pred >= 0.5).astype(int)
        acc = np.sum(y_test_pred == y_test) / len(y_test)
        logger.info(f"Test acc: {acc*100:.2f}%")
        test_metrics[0, test_metrics.shape[1] - 1] = epoch
        test_metrics[1, test_metrics.shape[1] - 1] = test_loss
        test_metrics[2, test_metrics.shape[1] - 1] = acc
    return net, train_metrics, test_metrics


class TorchModel:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("cbrec.torchmodel.TorchModel")
        self.net = None
        self.X = None
        self.y_true = None
        self.scaler = None
                
        # rec_input_matrix_cache is a map of metadata_id -> feature matrix
        # TODO should use a real cache, and probably add a flag to turn off prediction feature caching by default
        self.rec_input_matrix_cache = {}
        
    def set_training_data(self, X_train, y_train):
        self.X = X_train
        self.y_true = y_train
        # train scaler
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.logger.info(f"Scaler fit to training data with shape {self.X.shape}.")
    
    def train_model(self):
        X = self.X
        y_true = self.y_true
        n_train = int(np.ceil(len(y_true) * 0.99))
        X_train = X[:n_train,:]
        X_test = X[n_train:,:]
        y_train = y_true[:n_train]
        y_test = y_true[n_train:]
        self.logger.info(f"Train/validation sizes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")
        
        self.net, self.train_metrics, self.test_metrics = train_pytorch_model(X_train, y_train, X_test, y_test)
        self.logger.info("Finished training net.")
        
    def save_model_metrics(self, experiment_name=None, show_graph=False):
        assert self.train_metrics is not None and self.test_metrics is not None
                
        if experiment_name is None:
            experiment_name = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            
        this_torch_experiment_dir = os.path.join(self.config.torch_experiments_dir, experiment_name)
        os.makedirs(this_torch_experiment_dir, exist_ok=True)
        
        with open(os.path.join(this_torch_experiment_dir,"train_metrics.pkl"), 'wb') as file:
            pickle.dump(self.train_metrics, file)
            
        with open(os.path.join(this_torch_experiment_dir,"test_metrics.pkl"), 'wb') as file:
            pickle.dump(self.test_metrics, file)
        
        fig = plt.figure()
        plt.plot(self.train_metrics[0],self.train_metrics[1])             
        plt.plot(self.test_metrics[0],self.test_metrics[1])
        plt.legend(["Train","Test"])
        plt.title('Model Loss')
        plt.xlabel("Epoch")
        plt.savefig(os.path.join(this_torch_experiment_dir,"loss.png"))
        if show_graph:
            plt.show()
        plt.close(fig)

        
        fig = plt.figure()
        plt.plot(self.train_metrics[0],self.train_metrics[2])             
        plt.plot(self.test_metrics[0],self.test_metrics[2])
        plt.legend(["Train","Test"])
        plt.title('Model Accuracy')
        plt.xlabel("Epoch")
        plt.savefig(os.path.join(this_torch_experiment_dir,"accuracy.png"))
        if show_graph:
            plt.show()
        plt.close(fig)
    
    def test_model(self, test_md_list):
        """
        Adds metrics directly to the `md` metadata dictionaries contained in `test_md_list`.
        """
        return self.predict_from_test_contexts(test_md_list)
    
    def get_pointwise_training_triples(self):
        db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
        text_db = embeddingdb.get_text_feature_db(self.config)
        feature_arrs = []
        ys = []
        try:
            n_invalid = 0
            with open(os.path.join(self.config.model_data_dir, 'missing_train_journal_oids.txt'), 'w') as outfile:
                prev_size = 0  # tracks the size of self.missing_journal_id_list
                n_missed = 0
                for row in tqdm(featuredb.stream_triples(db), desc='Streaming training triples', total=300000):
                    try:
                        source_target_arr, source_alt_arr = self.get_input_arrs_from_triple_dict(row, text_db)
                    except:
                        n_invalid += 1
                        if len(self.missing_journal_id_list) > prev_size:
                            for journal_oid in self.missing_journal_id_list[prev_size:]:
                                outfile.write(journal_oid + "\n")
                                n_missed += 1
                            prev_size = len(self.missing_journal_id_list)
                        continue
                    feature_arrs.append(source_target_arr)
                    ys.append(1)
                    feature_arrs.append(source_alt_arr)
                    ys.append(0)
        finally:
            db.close()
            text_db.close()
            if n_invalid > 0:
                self.logger.warning(f"Excluded {n_invalid} invalid train triples. (Missed journals: {n_missed})")
        self.y_true = np.array(ys)
        self.X = np.vstack(feature_arrs)
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        return self.X, self.y_true
    
    def predict_from_test_contexts(self, test_md_list, site_allowlist=None):
        def is_usp_allowed(usp):
            if site_allowlist is None:
                return True
            else:
                return usp[1] in site_allowlist
        self.logger.info(f"Predicting for {len(test_md_list)} test contexts.")
        db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
        predictions = []
        try:
            for md in tqdm(test_md_list, desc="Predicting"):
                metadata_id = md['metadata_id']
                test_context = featuredb.get_test_context_by_metadata_id(db, metadata_id, self.config)
                # Verify and remove candidate USPs that don't have available texts
                # TODO remove this step once processing is fixed and the relevant assumption (that all candidate_usps are actually eligible) holds
                # should probably keep the site_allowlist though
                candidate_usp_arr = test_context['candidate_usp_arr'].astype(np.int64)
                candidate_usps = [(candidate_usp_arr[i,0], candidate_usp_arr[i,1]) for i in range(candidate_usp_arr.shape[0])]
                invalid_mask = np.array([not is_usp_allowed(usp) or len(self.journal_id_lookup.get_journal_updates_before(usp, md['timestamp'])) < 3 
                                         for usp in candidate_usps])
                if np.sum(invalid_mask) > 0:
                    self.logger.info(f"Removing {np.sum(invalid_mask)} / {len(candidate_usp_arr)} candidate USPs that are not allowed or don't have available texts.")
                    test_context['candidate_usp_arr'] = candidate_usp_arr[~invalid_mask]
                    test_context['candidate_usp_mat'] = test_context['candidate_usp_mat'][~invalid_mask]
                                   
                rc = reccontext.RecContext.create_from_test_context(self.config, md, test_context)
                try:
                    self.get_input_matrix_from_test_context(rc)  # force cache generation of the features for this rec context BEFORE diving into the scorer
                except Exception as ex:
                    self.logger.error(ex)
                    continue

                scorer = evaluation.TorchModelScorer(self.config, rc, self, "PointwiseLinearTorchModel", save_scores=True)
                metric_dict = scorer.score()
                md['baseline_metrics']['PointwiseLinearTorchModel'] = metric_dict
                
                predictions.append((rc, scorer))
        finally:
            db.close()
        return predictions
    
    
    def create_train_triples_from_test_contexts(self, test_md_list):
        """
        This function does two things:
         - Creates training triples by negative sampling from test context candidate USPs
         - Identifies journal updates needed to generate features for those training triples
        
        It returns the training triples such that they can be passed to self.get_input_arrs_from_triple_dict(d, text_db),
        where d must contain:
         - {source,target,alt}_user_id
         - {source,target,alt}_site_id
         - interaction_timestamp
         - {source,target,alt}_feature_arr
         - source_target_feature_arr
         - source_alt_feature_arr
        """
        self.logger.info(f"Creating training triples from {len(test_md_list)} test contexts.")
        triple_dicts = []
        required_journal_ids = set()  # set of the journal ids required to generate features for the created triples
        rng = np.random.default_rng(12)  # use a new default_rng instance to ensure the same alternatives will be selected when given the same set of test contexts
        db = featuredb.get_db_by_filepath(self.config.feature_db_filepath)
        with db:
            for md in tqdm(test_md_list, desc="Creating training triples"):
                metadata_id = md['metadata_id']
                test_context = featuredb.get_test_context_by_metadata_id(db, metadata_id, self.config)
                interaction_timestamp = int(md['timestamp'])
                
                source_usp_arr = test_context['source_usp_arr'].astype(np.int64)
                candidate_usp_arr = test_context['candidate_usp_arr'].astype(np.int64)
                
                source_usp_mat = test_context['candidate_usp_mat']
                candidate_usp_mat = test_context['candidate_usp_mat']
                user_pair_mat = test_context['user_pair_mat']
                
                target_inds = test_context['target_inds'].astype(np.int64)
                
                # Identify indices that are appropriate to use as alternatives
                # (which is any non-target)
                alt_candidate_mask = np.ones(len(candidate_usp_arr)).astype(bool)
                alt_candidate_mask[target_inds] = False
                alt_candidate_inds = np.arange(len(candidate_usp_arr))[alt_candidate_mask]
                
                for i, source_usp in enumerate(source_usp_arr):
                    source_journal_ids = self.journal_id_lookup.get_journal_updates_before(tuple(source_usp), interaction_timestamp)
                    if len(source_journal_ids) < 3:
                        self.logger.warning(f"Source USP {source_usp} has <3 journals available, despite being generated as test context {metadata_id}.")
                        continue
                    
                    # extract the source features for this USP
                    source_feature_arr = source_usp_mat[i,:]
                    
                    for target_ind in target_inds:
                        target_usp = candidate_usp_arr[target_ind]
                        
                        # Extract target arrs from among the candidates
                        target_feature_arr = candidate_usp_mat[target_ind,:]
                        user_pair_ind = (i * len(candidate_usp_arr)) + target_ind
                        source_target_feature_arr = user_pair_mat[user_pair_ind,:]

                        # Sample an alt from among the candidates
                        alt_ind = rng.choice(alt_candidate_inds)
                        alt_usp = candidate_usp_arr[alt_ind]
                        
                        # Extract alt arrs from among the candidates
                        alt_feature_arr = candidate_usp_mat[alt_ind,:]
                        user_pair_ind = (i * len(candidate_usp_arr)) + alt_ind
                        source_alt_feature_arr = user_pair_mat[user_pair_ind,:]
                        
                        # Identify target and alt required journal ids
                        target_journal_ids = self.journal_id_lookup.get_journal_updates_before(tuple(target_usp), interaction_timestamp)
                        if len(target_journal_ids) < 3:
                            self.logger.warning(f"Target USP {target_usp} has <3 journals available, despite being generated as a candidate in context {metadata_id}.")
                            continue
                        alt_journal_ids = self.journal_id_lookup.get_journal_updates_before(tuple(alt_usp), interaction_timestamp)
                        if len(alt_journal_ids) < 3:
                            self.logger.warning(f"Alt USP {alt_usp} has <3 journals available, despite being generated as a candidate in context {metadata_id}.")
                            continue
                        
                        d = {
                            'source_user_id': source_usp[0],
                            'source_site_id': source_usp[1],
                            'target_user_id': target_usp[0],
                            'target_site_id': target_usp[1],
                            'alt_user_id': alt_usp[0],
                            'alt_site_id': alt_usp[1],
                            'interaction_timestamp': interaction_timestamp,
                            'source_feature_arr': source_feature_arr,
                            'target_feature_arr': target_feature_arr,
                            'alt_feature_arr': alt_feature_arr,
                            'source_target_feature_arr': source_target_feature_arr,
                            'source_alt_feature_arr': source_alt_feature_arr,
                        }
                        triple_dicts.append(d)
                        required_journal_ids.update(source_journal_ids, target_journal_ids, alt_journal_ids)
        return triple_dicts, required_journal_ids
    
    
    def get_test2train_triples(self, triple_dicts):
        self.missing_journal_id_list = []  # reset the list of missing journal ids
        text_db = embeddingdb.get_text_feature_db(self.config)
        feature_arrs = []
        ys = []
        try:
            n_invalid = 0
            with open(os.path.join(self.config.model_data_dir, 'missing_test2train_journal_oids.txt'), 'w') as outfile:
                prev_size = 0  # tracks the size of self.missing_journal_id_list
                n_missed = 0
                for triple_dict in tqdm(triple_dicts, desc='Processing test2train triples'):
                    try:
                        source_target_arr, source_alt_arr = self.get_input_arrs_from_triple_dict(triple_dict, text_db)
                    except:
                        n_invalid += 1
                        if len(self.missing_journal_id_list) > prev_size:
                            for journal_oid in self.missing_journal_id_list[prev_size:]:
                                outfile.write(journal_oid + "\n")
                                n_missed += 1
                            prev_size = len(self.missing_journal_id_list)
                        continue
                    feature_arrs.append(source_target_arr)
                    ys.append(1)
                    feature_arrs.append(source_alt_arr)
                    ys.append(0)
        finally:
            text_db.close()
            if n_invalid > 0:
                self.logger.warning(f"Excluded {n_invalid} invalid test2train triples. (Missed journals: {n_missed})")
        y_true = np.array(ys)
        X = np.vstack(feature_arrs)
        X = self.scaler.transform(X)
        return X, y_true
    
    
    def create_input_arr(self, source_feature_arr, candidate_feature_arr, source_candidate_feature_arr):
        self.logger.warning("Call to deprecated create_input_arr")
        return np.concatenate([source_feature_arr, candidate_feature_arr, source_candidate_feature_arr])
    
    def get_input_arrs_from_triple_dict(self, triple_dict, text_db, include_text=True, record_missing_journal_ids=True):
        if include_text:
            source_usp = (triple_dict['source_user_id'], triple_dict['source_site_id'])
            target_usp = (triple_dict['target_user_id'], triple_dict['target_site_id'])
            alt_usp = (triple_dict['alt_user_id'], triple_dict['alt_site_id'])
            source_journal_ids = self.journal_id_lookup.get_journal_updates_before(source_usp, triple_dict['interaction_timestamp'])
            target_journal_ids = self.journal_id_lookup.get_journal_updates_before(target_usp, triple_dict['interaction_timestamp'])
            alt_journal_ids = self.journal_id_lookup.get_journal_updates_before(alt_usp, triple_dict['interaction_timestamp'])
            if len(source_journal_ids) < 3 or len(target_journal_ids) < 3 or len(alt_journal_ids) < 3:
                #self.logger.error(f"Insufficient texts: source n={len(source_journal_ids)}, target n={len(target_journal_ids)}, alt n={len(alt_journal_ids)}")
                raise ValueError(f"Insufficient texts: source n={len(source_journal_ids)}, target n={len(target_journal_ids)}, alt n={len(alt_journal_ids)}")
            source_text_arrs = embeddingdb.get_text_feature_arrs_from_db(text_db, source_journal_ids)
            target_text_arrs = embeddingdb.get_text_feature_arrs_from_db(text_db, target_journal_ids)
            alt_text_arrs = embeddingdb.get_text_feature_arrs_from_db(text_db, alt_journal_ids)
            if record_missing_journal_ids:
                for ids, texts in zip([source_journal_ids, target_journal_ids, alt_journal_ids], [source_text_arrs, target_text_arrs, alt_text_arrs]):
                    for journal_oid, text in zip(ids, texts):
                        if text is None:
                            self.missing_journal_id_list.append(journal_oid)
            source_text_arrs = [arr for arr in source_text_arrs if arr is not None]
            target_text_arrs = [arr for arr in target_text_arrs if arr is not None]
            alt_text_arrs = [arr for arr in alt_text_arrs if arr is not None]
            if len(source_text_arrs) < 3 or len(target_text_arrs) < 3 or len(alt_text_arrs) < 3:
                #self.logger.error(f"Embeddings unavailable for texts (total {len(self.missing_journal_id_list)}): source n={len(source_text_arrs)}, target n={len(target_text_arrs)}, alt n={len(alt_text_arrs)}")
                raise ValueError(f"Embeddings unavailable for texts (total {len(self.missing_journal_id_list)}): source n={len(source_text_arrs)}, target n={len(target_text_arrs)}, alt n={len(alt_text_arrs)}")
            source_text_arr = np.mean(source_text_arrs, axis=0)  # mean pool the available texts
            target_text_arr = np.mean(target_text_arrs, axis=0)
            alt_text_arr = np.mean(alt_text_arrs, axis=0)
            source_target_arr = np.concatenate([triple_dict['source_feature_arr'], triple_dict['target_feature_arr'], triple_dict['source_target_feature_arr'], source_text_arr, target_text_arr])
            source_alt_arr = np.concatenate([triple_dict['source_feature_arr'], triple_dict['alt_feature_arr'], triple_dict['source_alt_feature_arr'], source_text_arr, alt_text_arr])
        else:
            source_target_arr = self.create_input_arr(triple_dict['source_feature_arr'], triple_dict['target_feature_arr'], triple_dict['source_target_feature_arr'])
            source_alt_arr = self.create_input_arr(triple_dict['source_feature_arr'], triple_dict['alt_feature_arr'], triple_dict['source_alt_feature_arr'])
        return source_target_arr, source_alt_arr
    
    def get_text_arr(self, text_db, usp, timestamp):
        journal_ids = self.journal_id_lookup.get_journal_updates_before(usp, timestamp)
        if len(journal_ids) < 3:
            raise ValueError(f"Insufficient texts: {usp} n={len(journal_ids)}")
        text_arrs = embeddingdb.get_text_feature_arrs_from_db(text_db, journal_ids)
        text_arrs = [arr for arr in text_arrs if arr is not None]
        if len(text_arrs) < 1:  # TODO this allows us to make predictions even when all journals are not available
            raise ValueError(f"Embeddings unavailable for texts: {usp} n={len(text_arrs)}")
        text_arr = np.mean(text_arrs, axis=0)  # mean pool the texts
        return text_arr
    
    def get_input_matrix_from_test_context(self, rc):
        if rc.metadata_id in self.rec_input_matrix_cache:
            return self.rec_input_matrix_cache[rc.metadata_id]
        arrs = []
        text_db = embeddingdb.get_text_feature_db(self.config)
        with text_db:
            for i in range(len(rc.source_usp_mat)):
                source_feature_arr = rc.source_usp_mat[i,:]
                source_usp = (rc.source_usp_arr[i,0], rc.source_usp_arr[i,1])
                source_text_arr = self.get_text_arr(text_db, source_usp, rc.timestamp)
                for j in range(len(rc.candidate_usp_mat)):
                    candidate_feature_arr = rc.candidate_usp_mat[j,:]

                    ind = (i * len(rc.candidate_usp_arr)) + j
                    source_candidate_feature_arr = rc.user_pair_mat[ind,:]
                    
                    candidate_usp = (rc.candidate_usp_arr[j,0], rc.candidate_usp_arr[j,1])
                    candidate_text_arr = self.get_text_arr(text_db, candidate_usp, rc.timestamp)
                    
                    arr = np.concatenate([source_feature_arr, candidate_feature_arr, source_candidate_feature_arr, source_text_arr, candidate_text_arr])
                    arrs.append(arr)
        X = np.vstack(arrs)
        X = self.scaler.transform(X)
        self.rec_input_matrix_cache[rc.metadata_id] = X
        return X
        
    def score_test_matrix(self, X_test, should_rescale=True):
        if should_rescale:
            X_test = self.scaler.transform(X_test)
        self.net.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test)
            outputs = self.net(X_test_tensor)
            y_test_score = torch.sigmoid(outputs.detach()).view((-1,)).numpy()
            return y_test_score
