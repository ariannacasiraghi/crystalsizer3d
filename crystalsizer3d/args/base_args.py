from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, _ArgumentGroup
from typing import Union

from crystalsizer3d.util.utils import hash_data, to_dict


class BaseArgs(ABC): # ABC makes this an abstract base class, a template that I can’t use directly
    @classmethod
    @abstractmethod # This means that every subclass must write its own version
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        pass

    @classmethod
    def from_args(cls, args: Union[Namespace, dict]) -> 'BaseArgs': # args is the object from parser.parse_args() that holds the options for all groups
                                                                    # namespace is a simple object that argparse returns from parse_args()
                                                                    # If I type --n-epochs 5, then args.n_epochs is 5
        """
        Create a BaseArgs instance from command-line arguments.
        """
        if isinstance(args, dict):
            return cls(**args) # Unpack the dict into named arguments. E.g.,  {'a': 1, 'b': 2} becomes a=1, b=2
        elif isinstance(args, Namespace):
            return cls(**vars(args)) # Converts to a dictionary 
        else:
            raise ValueError(f'Unrecognised args type: {type(args)}')

    def clone(self) -> 'BaseArgs':
        """
        Clone the arguments into a new class.
        """
        return self.__class__(**self.to_dict())

    def to_dict(self) -> dict:
        """
        Convert the class to a dictionary.
        """
        return to_dict(self)

    def hash(self) -> str:
        """
        Get a hash of the class arguments.
        """
        return hash_data(self.to_dict())
