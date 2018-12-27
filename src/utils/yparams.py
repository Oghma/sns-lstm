"""Module that defines the class that load the hyperparameters from a yaml
file."""
import yaml
from tensorflow.contrib.training import HParams


class YParams(HParams):
    """Yparams load the parameters from a yaml file."""

    def __init__(self, file_name, config_name=None):
        """Constructor of the YParams class.

        Args:
          file_name: string. Path to the file containing the parameters.
          config_name: string. Name of the set of parameters. If None, the file
            has not sets.

        """
        super().__init__()

        with open(file_name) as fp:
            if config_name is not None:
                for k, v in yaml.load(fp)[config_name].items():
                    self.add_hparam(k, v)
            else:
                for k, v in yaml.load(fp).items():
                    self.add_hparam(k, v)
