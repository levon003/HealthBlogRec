
import os
import pickle
import logging
import numpy as np
from glob import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import cbrec.modeling.modelconfig
import cbrec.modeling.preprocess

import cbrec.modeling.models.linearnet
import cbrec.modeling.models.concatnet
import cbrec.modeling.models.simnet


class ModelTrainer:
    """
    Given a ModelConfig and data, instantiates and trains a model.
    
    Also exposes a function (score_matrix) for predictions.
    
    The feature_manager is used to instantiate nets that need to know about particular types of features.
    """
    def __init__(self, config: cbrec.modeling.modelconfig.ModelConfig, feature_manager: cbrec.modeling.preprocess.FeatureManager):
        self.config = config
        self.feature_manager = feature_manager
        
        self.net = None
        self.train_metrics = None
        self.test_metrics = None
        
        self.best_model_description = None  # string or None; if string, is the description needed to load the best model during training
    
    
    def create_net(self):
        if self.config.model_name == 'LinearNet':
            net = cbrec.modeling.models.linearnet.LinearNet(self.config)
        elif self.config.model_name == 'ConcatNet':
            net = cbrec.modeling.models.concatnet.ConcatNet(self.config, self.feature_manager)
        elif self.config.model_name == 'SimNet':
            net = cbrec.modeling.models.simnet.SimNet(self.config, self.feature_manager)
        elif self.config.model_name == 'LearnedSimNet':
            net = cbrec.modeling.models.learnedsimnet.LearnedSimNet(self.config, self.feature_manager)
        else:
            raise ValueError(f"Unknown model name '{self.config.model_name}'.")
        return net
    
    
    def create_scheduler(self, optimizer, n_minibatches, n_epochs):
        if self.config.train_scheduler_name == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.train_max_lr,
                steps_per_epoch=n_minibatches,
                epochs=n_epochs,
            )
            return scheduler
        elif self.config.train_scheduler_name == 'None':
            return None
        else:
            raise ValueError(f"Unknown scheduler name '{self.config.train_scheduler_name}'.")
    

    def train_model(self, X_train, y_train, X_test, y_test):
        logger = logging.getLogger("cbrec.modeling.train.train_model")

        n_train = len(y_train)
        n_test = len(y_test)
        
        minibatch_size = n_train
        if hasattr(self.config, "minibatch_size"):
            minibatch_size = self.config.minibatch_size
            logger.info(f"Using minibatch size {minibatch_size}.")
        minibatch_size = min(n_train, minibatch_size)  # if minibatch_size is larger than n_train, force it to n_train
        n_minibatches = int(np.ceil(n_train / minibatch_size))
        # note: input dim is 27 for non-text features + 768 for text features
        self.config.LinearNet_n_input = X_train.shape[1]
        self.config.ConcatNet_n_input = X_train.shape[1]
        # create the net
        net = self.create_net()
        
        n_epochs = self.config.train_n_epochs
        criterion = nn.BCEWithLogitsLoss()  # pointwise loss function

        X_test_tensor = torch.from_numpy(X_test)
        y_test_tensor = torch.from_numpy(y_test)
        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(y_train)
        y_train_tensor = y_train_tensor.view(-1, 1)  # make labels 2-dimensional
        y_train_tensor = y_train_tensor.type_as(X_train_tensor)
        if self.config.train_verbose:
            logger.info(f"Input tensor sizes: {X_train_tensor.size()}, {y_train_tensor.size()}")
            logger.info(f"Validating model every {int(1/self.config.train_validation_rate)} epochs for {n_epochs} epochs.")

        # _metrics[0] -> Epoch, metrics[1] -> loss, _metrics[2] -> accuracy
        test_metrics = np.zeros((3,int(n_epochs*self.config.train_validation_rate+1))) #+1 to ensure space for final epoch metric
        train_metrics = np.zeros((3,n_epochs))

        if n_epochs > 0:
            #optimizer = optim.SGD(net.parameters(), lr=lr_init, momentum=0.9)
            optimizer = optim.Adam(net.parameters(),
                                   lr=self.config.train_lr_init,
                                   betas=(self.config.train_Adam_beta1, self.config.train_Adam_beta2),
                                   eps=self.config.train_Adam_eps,
                                   weight_decay=self.config.train_weight_decay)
            scheduler = self.create_scheduler(optimizer, n_minibatches, n_epochs)
        else:
            epoch = 0
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
                if self.config.train_verbose and epoch == 0:
                    logger.info(f"    Minibatch for inds in {minibatch_start} - {minibatch_end}.")
                minibatch_inds = epoch_order[minibatch_start:minibatch_end]

                inputs = X_train_tensor[minibatch_inds]
                train_labels = y_train_tensor[minibatch_inds]

                net.train()
                train_outputs = net(inputs)
                train_loss = criterion(train_outputs, train_labels)
                train_loss.backward()
                optimizer.step()
                if scheduler is not None:
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
            if self.config.train_verbose and (epoch < 3 or epoch == n_epochs - 1 or epoch % 10 == 0 or should_stop_early):
                logger.info(f"{epoch:>3} ({datetime.now() - s}): train loss={train_loss:.4f} train accuracy={train_acc*100:.2f}% LR={optimizer.param_groups[0]['lr']:.2E}")
            if should_stop_early:
                break

            if epoch % (1/self.config.train_validation_rate) == 0:
                net.eval()
                with torch.no_grad():
                    test_outputs = net(X_test_tensor)
                    test_loss = criterion(test_outputs.detach(), y_test_tensor.unsqueeze(1).float())
                    y_test_pred = torch.sigmoid(test_outputs.detach()).view((-1,)).numpy()
                    y_test_pred = (y_test_pred >= 0.5).astype(int)
                    test_acc = np.sum(y_test_pred == y_test) / len(y_test)
                logger.info(f"    {epoch:>3}: test loss={test_loss:.4f} test accuracy={test_acc*100:.2f}%")
                metric_ind = int(epoch*self.config.train_validation_rate)
                if metric_ind > 0 and test_loss <= np.min(test_metrics[1,:metric_ind]):
                    # this is the lowest loss we've reached
                    self.net = net
                    self.save_model_state_dict(description=f"e{epoch}")
                    self.best_model_description = f"e{epoch}"
                    logger.info(f"    Best validation lost achieved so far. Model checkpoint saved.")
                # TODO consider an else clause here to terminate if enough epochs have passed without improving validation loss
                test_metrics[0,metric_ind] = epoch
                test_metrics[1,metric_ind] = test_loss
                test_metrics[2,metric_ind] = test_acc

        if self.config.train_verbose and n_epochs > 0:
            final_train_loss = train_loss
            final_epoch_count = epoch + 1
            logger.info(f"Completed {final_epoch_count} epochs with a final train loss of {final_train_loss:.4f}.")

        net.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test)
            outputs = net(X_test_tensor)
            test_loss = criterion(outputs.detach(), y_test_tensor.unsqueeze(1).float())
            y_test_pred = torch.sigmoid(outputs.detach()).view((-1,)).numpy()
            y_test_pred = (y_test_pred >= 0.5).astype(int)
            acc = np.sum(y_test_pred == y_test) / len(y_test)
            logger.info(f"Test acc: {acc*100:.2f}%")
            test_metrics[0, test_metrics.shape[1] - 1] = epoch
            test_metrics[1, test_metrics.shape[1] - 1] = test_loss
            test_metrics[2, test_metrics.shape[1] - 1] = acc
            
        self.net = net
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        return net, train_metrics, test_metrics
    
    
    def score_matrix(self, X):
        self.net.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X)
            outputs = self.net(X_tensor)
            y_score = torch.sigmoid(outputs.detach()).view((-1,)).numpy()
            return y_score
    
    
    def get_train_metrics(self):
        return self.train_metrics, self.test_metrics
            
    
    def load_model_state_dict(self, description=None):
        """
        Depends on output_basename and output_dir being set in the config.
        """
        logger = logging.getLogger("cbrec.modeling.train.ModelTrainer.load_model_state_dict")
        
        if description is None:
            state_dict_filepath = os.path.join(self.config.output_dir, self.config.output_basename + '.pt')
        else:
            state_dict_filepath = os.path.join(self.config.output_dir, self.config.output_basename + f'_{description}.pt')
        if not os.path.exists(state_dict_filepath):
            raise ValueError(f"No state dict found at expected path '{state_dict_filepath}'.")
        net = self.create_net()
        net.load_state_dict(torch.load(state_dict_filepath), strict=False)
        self.net = net
        logger.info("Instantiated model and loaded state dict.")
        
        
    def load_train_metrics(self):
        train_metrics_filepath = os.path.join(self.config.output_dir, self.config.output_basename + '_train_metrics.pkl')
        if os.path.exists(train_metrics_filepath):
            with open(train_metrics_filepath, 'rb') as infile:
                self.train_metrics = pickle.load(infile)
        
        valid_metrics_filepath = os.path.join(self.config.output_dir, self.config.output_basename + '_valid_metrics.pkl')
        if os.path.exists(valid_metrics_filepath):
            with open(valid_metrics_filepath, 'rb') as infile:
                self.test_metrics = pickle.load(infile)
                
    
    def save_model_state_dict(self, description=None):
        if self.net is not None:
            if description is None:
                state_dict_filepath = os.path.join(self.config.output_dir, self.config.output_basename + '.pt')
            else:
                state_dict_filepath = os.path.join(self.config.output_dir, self.config.output_basename + f'_{description}.pt')
            torch.save(self.net.state_dict(), state_dict_filepath)
        else:
            state_dict_filepath = None
        return state_dict_filepath
    

    def save_train_metrics(self):
        """
        Saves the metrics produced during training.
        
        Currently stores self.train_metrics and self.test_metrics.
        """
        logger = logging.getLogger("cbrec.modeling.train.ModelTrainer.save_train_metrics")
        
        # train metrics
        if self.train_metrics is not None:
            train_metrics_filepath = os.path.join(self.config.output_dir, self.config.output_basename + '_train_metrics.pkl')
            with open(train_metrics_filepath, 'wb') as outfile:
                pickle.dump(self.train_metrics, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            train_metrics_filepath = None
            self.logger.info("No train training metrics to save.")
        
        # validation metrics
        if self.test_metrics is not None:
            valid_metrics_filepath = os.path.join(self.config.output_dir, self.config.output_basename + '_valid_metrics.pkl')
            with open(valid_metrics_filepath, 'wb') as outfile:
                pickle.dump(self.test_metrics, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            valid_metrics_filepath = None
            self.logger.info("No validation training metrics to save.")
        return train_metrics_filepath, valid_metrics_filepath
    
    
    def save_model_metrics(self, root_dir, experiment_name=None, show_graph=False):
        """
        Deprecated.
        
        TODO probably delete this function, no real need to save the figures when we have the raw data to generate them saved already.
        """
        assert self.train_metrics is not None and self.test_metrics is not None
                
        if experiment_name is None:
            experiment_name = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            
        this_torch_experiment_dir = os.path.join(root_dir, experiment_name)
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
