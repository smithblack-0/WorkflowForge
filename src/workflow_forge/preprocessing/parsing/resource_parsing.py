"""
Resource parsing logic parses, naturally,
something and turns them into resources
"""
import os
from ..resources import AbstractResource, ListSamplerResource, StringResource
from typing import Dict, List, Tuple

def parse_constitution(file_path: str)->Dict[str, AbstractResource]:
    """
    Parses a single constitution file into a dictionary
    of resources. Splits on [Point] tokens into pieces.
    :param file_path: The file_path to load
    :return: The dictionary of resources parsed
    """
    if not file_path.endswith('.txt'):
        raise ValueError('filename must end with .txt')
    with open(file_path, 'r') as f:
        text = f.read()
    pieces = text.split("[Point]")
    name = os.path.basename(file_path).split('.')[0]
    output = {f"{name}_overview": StringResource(pieces[0])}
    if len(pieces) > 1:
        string_list = pieces[1:]
        string_list = [f"{i})" + item for i, item in enumerate(string_list)]
        output[f"{name}_details"] = ListSamplerResource(string_list),
    return output

def parse_constitutions(directory_path: str)->Dict[str, AbstractResource]:
    """
    Parses a directory of constitution files into a resource dictionary
    :param directory_path: The path for the directory
    :return: The dictionary of resources parsed
    """
    files = os.listdir(directory_path)
    output = {}
    for file in files:
        if file.endswith('.txt'):
            path = os.path.join(directory_path, file)
            output.update(parse_constitution(path))
    return output

