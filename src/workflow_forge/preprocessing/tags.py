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
import torch
from torch import nn
from typing import Dict, List

class TagConverter(nn.Module):
    """
    A tag converter may accept a list of tags and
    returns a boolean flag vector where the
    active tags are marked as on, and the
    rest are marked as off.
    """
    @property
    def device(self) -> torch.device:
        return self.device_tracker.device
    @property
    def num_tags(self)->int:
        return len(self.tag_conversions)

    def __init__(self,
                 tag_conversions: Dict[str, int]
                 ):
        """
        The initialization of the tag converter
        :param tags: A one-to-one mapping of tag name
            to int value
        """
        super().__init__()
        self.device_tracker = nn.Parameter(torch.tensor(1.0)) # Used just to track device.
        self.tag_conversions = tag_conversions
        self.inverse_conversions = {v : k for k, v in tag_conversions.items()}

    def tensorize(self, tags: List[str])->torch.Tensor:
        """
        Converts a list of tags into its tensor equivalent
        :param tags: The list of tags
        :return: The boolean tensor array. True means tag was present
        """
        output = torch.zeros(self.num_tags, dtype=torch.bool, device=self.device)
        for tag in tags:
            idx = self.tag_conversions[tag]
            output[idx] = True
        return output

    def detensorize(self, tags: torch.Tensor)->List[str]:
        """
        Converts a tensor tag bool array into its list equivalent.
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



class TagConverterBuilder:
    """
    A specialized tag convert builder, it is able
    to accept collections of tag in sequence
    in order to find the set of unique tags that exist.
    Once all tags have been poured into it, the
    .build method can be called to make a tag converter
    """
    def __init__(self):
        self.tag_set = set()
    def __call__(self, tags: List[str]):
        """
        Add a tag collection to the set.
        :param tags: The tags to add
        """
        tags = set(tags)
        self.tag_set.update(tags)
    def build(self)->TagConverter:
        """
        Builds the tag converter now that we
        have seen the entire tag collection.
        :return: The tag converter object
        """
        tags = sorted(list(self.tag_set))
        convertion_dict = {tag : i for i, tag in enumerate(tags)}
        return TagConverter(convertion_dict)