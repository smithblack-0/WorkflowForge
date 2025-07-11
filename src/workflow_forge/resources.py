"""
A resource is a way to resolve a formatting dependency.
It consists of a callable object which can produce
a single string output and accept arguments. The exact
nature of the accepted arguments varies based on
configuration, as does the underlying sampling
mechanism.
"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class AbstractResource(ABC):
    """
    The abstract resource class. Specifies
    the usage contract
    """
    @abstractmethod
    def __call__(self, **kwargs)->str:
        """
        Must return a string of some sort to resolve
        the dynamic dependency
        :param kwargs: The keyword arguments
        :return: A string
        """

class StaticStringResource(AbstractResource):
    """
    A simple string resource. It returns
    exactly what the resource was initialized
    with
    """
    def __init__(self, string: str):
        self.string = string
    def __call__(self)->str:
        return self.string


class ListSamplerResource(AbstractResource):
    """
    A string sampler resource is capable of drawing
    samples randomly from among an internal
    list of strings without replacement (exhaustion).
    """

    def __init__(self, string_list: List[str]):
        assert len(string_list) > 0
        self.string_list = string_list
        self.remaining_items = string_list.copy()

    def __call__(self, num_samples: Union[int, str]) -> str:
        output = []
        if isinstance(num_samples, int):
            # If we don't have enough remaining items, reset the pool
            if len(self.remaining_items) < num_samples:
                self.remaining_items = self.string_list.copy()

            # Sample without replacement
            sampled_indices = np.random.choice(
                len(self.remaining_items),
                size=min(num_samples, len(self.remaining_items)),
                replace=False
            )

            # Get the sampled items and remove them from remaining_items
            for idx in sorted(sampled_indices, reverse=True):
                output.append(self.remaining_items.pop(idx))

        elif isinstance(num_samples, str):
            if num_samples == "all":
                output = self.string_list
            else:
                raise NotImplementedError("Unknown string sampling type")
        return "\n".join(output)

class LRUBufferResource(ListSamplerResource):
    """
    A specialized list sampler that maintains an
    internal repository of examples to sample from
    and can add new ones. If the buffer is full,
    the oldest strings are removed to make room.
    """
    def __init__(self, buffer_size: int):
        self.buffer = [""]
        super().__init__(self.buffer)
        self.buffer_size = buffer_size
    def good_synthetic_training_data(self, string: str):
        """
        Insert a new string into the buffer
        :param string: The buffer to insert
        """
        self.buffer.insert(0, string)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(-1)
