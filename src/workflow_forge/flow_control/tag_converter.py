"""
The tagging system has to be able to construct
preprocessing tag converters, and then later on convert tag collections
for the three kinds of spans into bool flag collections. These
functionalities are centralized here.

Tag conversion is complicated by the fact that the full conversion
dictionary is not known until parsing of the entire YAML dictionary
is done, since that is where we discover what tags are in use. As a
result, conversion must be delayed.

We use a two-stage flow involving capturing the tags in use, then
building an object that can convert tag collections when all tags
are captured.
"""
import numpy as np
from typing import Dict, List

class TagConverter:
    """
    A tag converter may accept a list of tags and
    returns a boolean flag vector where the
    active tags are marked as on, and the
    rest are marked as off.
    """
    @property
    def num_tags(self) -> int:
        return len(self.tag_conversions)

    def __init__(self,
                 tag_conversions: Dict[str, int]
                 ):
        """
        The initialization of the tag converter
        :param tag_conversions: A one-to-one mapping of tag name
            to int value
        """
        self.tag_conversions = tag_conversions
        self.inverse_conversions = {v : k for k, v in tag_conversions.items()}

    def tensorize(self, tags: List[str]) -> np.ndarray:
        """
        Converts a list of tags into its array equivalent
        :param tags: The list of tags
        :return: The boolean array. True means tag was present
        """
        output = np.zeros(self.num_tags, dtype=np.bool_)
        for tag in tags:
            idx = self.tag_conversions[tag]
            output[idx] = True
        return output

    def detensorize(self, tags: np.ndarray) -> List[str]:
        """
        Converts a tag bool array into its list equivalent.
        :param tags: 1d bool array
        :return: The tag equivalent
        """
        assert tags.ndim == 1
        assert tags.shape[-1] == self.num_tags
        output = []
        for i, tag in enumerate(tags):
            if tag:
                output.append(self.inverse_conversions[i])
        return output