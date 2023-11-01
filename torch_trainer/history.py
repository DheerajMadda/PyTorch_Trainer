import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.basicConfig(
    format='[%(levelname)s] : %(message)s',
    level=logging.INFO, 
    stream=sys.stdout
)
log = logging.getLogger()

class History:
    """
    This class records the history
    """
    def __init__(self):
        """
        This is the initialization method
        """
        self._history = {
            "loss" : [],
            "val_loss" : [],
            "lr" : [],
            "num_epochs" : 0
        }
        self.is_validation = True

    @property
    def _return_history(self):
        """
        This method returns the history by removing "val_" keys
        if validation was not available

        Returns
        -------
        dict

        """

        if self.is_validation:
            return self._history
        else:
            history = {}
            for key in self._history:
                if "val_" not in key:
                    history[key] = self._history[key]
            return history

    def __str__(self):
        """
        This method returns the serialized history

        Returns
        -------
        str

        """
        return json.dumps(self._return_history, indent=4)
    
    def keys(self):
        """
        This method returns the keys history

        Returns
        -------
        list

        """
        return list(self._return_history.keys())
    
    def __getitem__(self, key):
        """
        This method returns the item value

        Parameters
        ----------
        key : str
            A string representing the history key

        Returns
        -------
        object

        """
        try:
            return self._history[key]
        except:
            raise KeyError("Provided key is not available.")
    
    def __setitem__(self, key, value):
        """
        This method assigns value to the respective key

        Parameters
        ----------
        key : str
            A string representing the history key
        value : object
            An object representing the history key's value

        Returns
        -------
        None

        """
        self._history[key] = value

    def __add__(self, value):
        """
        This method adds the value to the history's num_epochs

        Parameters
        ----------
        value : object
            An object representing the value

        Returns
        -------
        int

        """
        return self._history["num_epochs"] + value
    
    def __iter__(self):
        """
        This method makes the instance iterable

        Returns
        -------
        object

        """
        return iter(self._history.keys())

    def to_dict(self):
        """
        This method returns the history result

        Returns
        -------
        dict

        """
        return self._return_history

    def to_pandas(self):
        """
        This method creates a pandas datafrom from the history result

        Returns
        -------
        object

        """
        history = {}
        recorded_history = self._return_history
        history["Epoch"] = list(range(1, recorded_history["num_epochs"] + 1))

        for key in recorded_history:
            if key != "num_epochs":
                history[key.title()] = recorded_history[key]

        columns = list(history.keys())
        df = pd.DataFrame(history, columns=columns)
        return df

    def to_csv(self, save_path=None):
        """
        This method saves the history

        Parameters
        ----------
        save_path : str
            A file path to save history
            
        Returns
        -------
        None
        
        """
        if save_path is None:
            raise Exception("`save_path` must be provided!")
        
        df = self.to_pandas()
        df.to_csv(save_path, encoding='utf-8', header=True, index=False)

        log.info(f"History is saved as csv at path: {save_path}")

    def save_history(self, save_path=None, history=None):
        """
        This method saves the history

        Parameters
        ----------
        save_path : str
            A file path to save history
        history : dict
            A history dictionary
            
        Returns
        -------
        None
        
        """

        if save_path is None:
            raise Exception("`save_path` must be provided!")

        if history is None:
            history = self._history
        elif not isinstance(history, (dict,)):
            raise Exception("history must be one of type dict")

        with open(save_path, 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        log.info(f"History is saved at path: {save_path}")

    def load_history(self, load_path=None, history=None):
        """
        This method saves the history

        Parameters
        ----------
        load_path : str
            A file path to load history from
        history : dict
            A history dictionary
            
        Returns
        -------
        None
        
        """
        if load_path is None and history is None:
            raise Exception("Either pass `load_path` (str) or `history` (dict) as an argument")
        elif history:
            if not isinstance(history, (dict,)):
                raise Exception("history must be one of type dict")
            else:
                if len(history) == 0:
                    raise Exception("history must not be an empty dictionary")

        if load_path:
            with open(load_path, 'rb') as handle:
                history = pickle.load(handle)

        self._history = history

    def _type_validation(self, _type):
        """
        This method validates the type of the loss and metric

        Parameters
        ----------
        _type : str | None
            A string representing the type of the loss/metric
            Should be one of ('train', 'val')
            
        Returns
        -------
        None
        
        """
        if _type:
            if not isinstance(_type, (str,)) or _type.lower() not in ("train", "val"):
                raise Exception("_type should be of type `str` and be one of ('train', 'val')")
            
        elif _type == "val" and not self.is_validation:
            raise Exception("Validation history is not available.")
            
    def plot_loss(self, history=None, _type=None, save_path=None):
        """
        This method plots the loss

        Parameters
        ----------
        history : dict | None
            A history dictionary or `None`
        _type : str | None
            A string representing the type of the loss/metric
            Should be one of ('train', 'val')
        save_path : str
            A string that represents the path to which the plot to be saved
            
        Returns
        -------
        None
        
        """
        
        if history is None:
            history = self._history
        elif not isinstance(history, (dict,)):
            raise Exception("history must be one of type dict")
        
        self._type_validation(_type)
        epochs = range(history['num_epochs'])

        if _type == "train":
            plt.plot(epochs, history['loss'], 'r', label='Training loss')
        if _type == "val":
            plt.plot(epochs, history['val_loss'], 'b', label='Validation loss')
        if _type is None:
            plt.plot(epochs, history['loss'], 'r', label='Training loss')
            if self.is_validation:
                plt.plot(epochs, history['val_loss'], 'b', label='Validation loss')

        plt.title('Loss')
        plt.legend(loc=0)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_metric(self, name, history=None, _type=None, save_path=None):
        """
        This method plots the metric

        Parameters
        ----------
        name : str
            A string specifying the name of the metric
        history : dict | None
            A history dictionary or `None`
        _type : str | None
            A string representing the type of the loss/metric
            Should be one of ('train', 'val')
        save_path : str
            A string that represents the path to which the plot to be saved
            
        Returns
        -------
        None
        
        """
        if history is None:
            history = self._history
        elif not isinstance(history, (dict,)):
            raise Exception("history must be one of type dict")
        
        exclude_set = {"loss", "val_loss", "lr", "num_epochs"}
        if name in exclude_set:
            raise Exception("Provided name is not a metric.")
        if name not in history.keys():
            raise Exception("Provided name is not available in history.")
        
        self._type_validation(_type)
        epochs = range(history['num_epochs'])

        if _type == "train":
            plt.plot(epochs, history[name], 'r', label=f'Training {name.title()}')
        if _type == "val":
            plt.plot(epochs, history[f'val_{name}'], 'b', label=f'Validation {name.title()}')
        if _type is None:
            plt.plot(epochs, history[name], 'r', label=f'Training {name.title()}')
            if self.is_validation:
                plt.plot(epochs, history[f'val_{name}'], 'b', label=f'Validation {name.title()}')

        plt.title(name.title())
        plt.legend(loc=0)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_metrics(self, history=None, _type=None, save_path=None):
        """
        This method plots all the metrics

        Parameters
        ----------
        history : dict | None
            A history dictionary or `None`
        _type : str | None
            A string representing the type of the loss/metric
            Should be one of ('train', 'val')
        save_path : str
            A string that represents the path to which the plot to be saved
            
        Returns
        -------
        None
        
        """
        if history is None:
            history = self._history
        elif not isinstance(history, (dict,)):
            raise Exception("history must be one of type dict")
        
        self._type_validation(_type)

        exclude_set = {"loss", "val_loss", "lr", "num_epochs"}
        history_set = set(history.keys())
        metrics = history_set - exclude_set
        val_metrics = {metric for metric in metrics if "val_" in metric}
        metrics = sorted(list(metrics - val_metrics))
        len_metrics = len(metrics)
        if len_metrics == 0:
            raise Exception("Metrics are not available in history.")
        
        epochs = range(history['num_epochs'])
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 6))
        if not isinstance(axes, (np.ndarray,)):
            axes = [axes]
        fig.tight_layout(pad=3.0)

        for idx, name in enumerate(metrics):

            if _type == "train":
                axes[idx].plot(epochs, history[name], 'r', label=f'Training {name.title()}')
            if _type == "val":
                axes[idx].plot(epochs, history[f'val_{name}'], 'b', label=f'Validation {name.title()}')
            if _type is None:
                axes[idx].plot(epochs, history[name], 'r', label=f'Training {name.title()}')
                if self.is_validation:
                    axes[idx].plot(epochs, history[f'val_{name}'], 'b', label=f'Validation {name.title()}')

            axes[idx].set_title(name.title())
            axes[idx].legend(loc=0)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
