import os
import sys
import logging

import torch
import numpy as np

logging.basicConfig(
    format='[%(levelname)s] : %(message)s',
    level=logging.INFO, 
    stream=sys.stdout
)
log = logging.getLogger()

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        This is the initialization method

        Parameters
        ----------
        patience : int
            Number of times to wait after last time validation loss improved
        delta : int
            Minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_current_checkpoint = True
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        """
        This method checks conditions for early stopping

        Parameters
        ----------
        val_loss : float
            A value representing the epoch validation loss

        Returns
        -------
        None
        
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.save_current_checkpoint = False
            if self.verbose:
                log.info(f'[EarlyStopping] Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
            self.save_current_checkpoint = True

class ModelCheckpoint:
    def __init__(
        self,
        root_dir="experiments",
        name="my_model",
        save_best_only=False, 
        save_model_only=False
    ):
        """
        This is the initialization method

        Parameters
        ----------
        root_dir : str
            The root directory of the checkpoint
        name : str
            The directory name of the checkpoint
        save_best_only : bool
            A boolean that decides to save only the best model
        save_model_only : bool
            A boolean that decides to save only the model
        
        """
        self.root_dir = root_dir
        self.name = name
        self.save_best_only = save_best_only
        self.save_model_only = save_model_only
        self.prev_best_loss = None
        
    def _initiate(self):
        """
        This method is to perform the checkpoint prerequisites 

        Returns
        -------
        None
        
        """
        if self.name.startswith("./") and self.name.endswith("/"):
            self.name = self.name[2:-1]
        elif self.name.startswith("./"):
            self.name = self.name[2:]
        elif self.name.endswith("/"):
            self.name = self.name[:-1]
            
        checkpoint_path = os.path.join(os.getcwd(), self.root_dir, self.name)
        os.makedirs(checkpoint_path, exist_ok=True)
        dirs = os.listdir(checkpoint_path)
        if len(dirs):
            previous_run = int(sorted(dirs, key=lambda x:int(x.split("_")[-1]), reverse=True)[0].split("_")[-1])
            new_run = previous_run + 1
        else:
            new_run = 0
        self.checkpoint_run_path = os.path.join(checkpoint_path, f"runs_{new_run}")
        os.makedirs(self.checkpoint_run_path, exist_ok=True)
        log.info(f"Checkpoint directory is created at location:- {self.checkpoint_run_path}")

    def _delete_prev_files(self, file_path):
        """
        This method deletes all the previous model files

        Parameters
        ----------
        file_path : str
            The file name to save the model

        Returns
        -------
        None
        
        """
        files = os.listdir(self.checkpoint_run_path)
        current_epoch = int(file_path[file_path.find("_E")+2: file_path.find("_L")])
        files_to_delete = list(filter(lambda x: int(x[x.find("_E")+2: x.find("_L")]) < current_epoch, files))
        for file_to_delete in files_to_delete: 
            os.remove(os.path.join(self.checkpoint_run_path, file_to_delete))

    def save_checkpoint(self, save_path, model, optimizer, scheduler):
        """
        This method saves the checkpoint

        Parameters
        ----------
        save_path : str
            The path to which the model to be saved
        model : object
            A torch.nn.Module object
        optimizer : object | None
            An optimizer
        scheduler : object | None
            A torch.optim.lr_scheduler object

        Returns
        -------
        None
        
        """
        if hasattr(model, "_orig_mod"):
            # a torch.compile model
            model_state_dict = model._orig_mod.state_dict()
        else:
            model_state_dict = model.state_dict()

        if self.save_model_only:
            checkpoint = model_state_dict
        else:
            checkpoint = {
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict()
            }
            if scheduler:
                checkpoint['scheduler'] = scheduler.state_dict()
            if not save_path.endswith(".tar"):
                save_path = save_path + ".tar"
                
        torch.save(checkpoint, save_path)

    def __call__(self, file_path, current_loss, model, optimizer=None, scheduler=None):
        """
        This method checks conditions for saving the checkpoints

        Parameters
        ----------
        file_path : str
            The file path to save the model and/or optimizer and/or scheduler
        current_loss : float
            Loss value for the current epoch
        model : object
            A torch.nn.Module object
        optimizer : object | None
            An optimizer
        scheduler : object | None
            A torch.optim.lr_scheduler object

        Returns
        -------
        None
        
        """
        save_path = os.path.join(self.checkpoint_run_path, file_path)

        if self.save_best_only:
            if self.prev_best_loss is None:
                self.prev_best_loss = current_loss
                self.save_checkpoint(save_path, model, optimizer, scheduler)
            else:
                if current_loss < self.prev_best_loss:
                    self.prev_best_loss = current_loss
                    self.save_checkpoint(save_path, model, optimizer, scheduler)
                    self._delete_prev_files(file_path)

        else:
            self.save_checkpoint(save_path, model, optimizer, scheduler)
