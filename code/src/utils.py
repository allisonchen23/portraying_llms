import json
import os, shutil
# import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
from PIL import Image
import pickle
from datetime import datetime
import subprocess

def read_file(filepath):
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.pickle') or filepath.endswith('.pkl'):
        return read_pickle(filepath)
    elif filepath.endswith('.json'):
        return read_json(filepath)
    elif filepath.endswith('.txt'):
        return read_lists(filepath)
    # elif filepath.endswith('.pth'):
    #     return torch.load(filepath)
    elif filepath.endswith('.png') or filepath.endswith('.jpg') or filepath.endswith('.jpeg'):
        return load_image(filepath)
    elif filepath.endswith('.npy'):
        return np.load(filepath)
    else:
        raise ValueError("File type '.{}' not supported".format(filepath.split('.')[-1]))
        
def write_file(data, filepath, overwrite=False, quiet=False, save_index=False):
    if overwrite or not os.path.exists(filepath):
        if filepath.endswith('.csv'):
            data.to_csv(filepath, index=save_index)
        elif filepath.endswith('.pickle'):
            write_pickle(data, filepath)
        elif filepath.endswith('.json'):
            write_json(data, filepath)
        elif filepath.endswith('.txt'):
            write_lists(data, filepath)
        # elif filepath.endswith('.pth'):
        #     torch.save(data, filepath)
        elif filepath.endswith('.png') or filepath.endswith('.jpg') or filepath.endswith('.jpeg'):
            save_image(data, filepath)
        elif filepath.endswith('.npy'):
            np.save(filepath, data, allow_pickle=False)
        else:
            raise ValueError("File type '.{}' not supported".format(filepath.split('.')[-1]))
        if not quiet:
            print("Saved file to {}".format(filepath))
    else:
        print("File at {} exists and not overwriting".format(filepath))
            
def read_lists(filepath):
    '''
    Stores a depth map into an image (16 bit PNG)
    Arg(s):
        filepath : str
            path to file where data will be stored
    '''

    text_list = []
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                text_list.append(line.rstrip())

    return text_list


def write_lists(texts, filepath):
    '''
    Stores line delimited paths into file
    Arg(s):
        filepath : str
            path to file to save list
        texts : list[str]
            texts to write into file
    '''

    with open(filepath, 'w') as o:
        for idx, text in enumerate(texts):
            # if integer, convert to string to write
            if type(text) == int:
                text = str(text)
            o.write(text + '\n')

# def append_lists(filepath, paths):
#     '''
#     Stores line delimited paths into file
#     Arg(s):
#         filepath : str
#             path to file to save paths
#         paths : list[str]
#             paths to write into file
#     '''

#     with open(filepath, 'w+') as o:
#         for idx in range(len(paths)):
#             o.write(paths[idx] + '\n')

def read_pickle(filepath):
    '''
    Return unserialized pickle object at filepath
    Arg(s):
        filepath : str
            path to pickle file

    Returns:
        object
    '''

    with open(filepath, 'rb') as f:
        return pickle.load(f)

def write_pickle(object, filepath):
    '''
    Serialize object as pickle file at filepath
    Arg(s):
        object : any
            Serializable object
        filepath : str
            file to write to
    '''
    # Create directory if it doesn't exist
    if not os.path.isdir(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    with open(filepath, 'wb') as f:
        pickle.dump(object, f)

def load_image(image_path, data_format='HWC', resize=None):
    '''
    Load image and return as CHW np.array

    Arg(s):
        image_path : str
            path to find image
        data_format : str
            order of channels to return
        resize : tuple(int, int) or None
            the resized shape (H, W) or None

    Returns :
        C x H x W np.array normalized between [0, 1]
        or 
        H x W x C np.array normalized between [0, 1]
    '''
    image = Image.open(image_path).convert("RGB")
    if resize is not None:
        image = image.resize(resize)
    # Convert to numpy array
    image = np.asarray(image, float)

    # Normalize between [0, 1]
    image = image / 255.0

    if data_format == "HWC":
        return image.astype(np.float32)
    elif data_format == "CHW":
        # Make channels C x H x W
        image = np.transpose(image, (2, 0, 1))
        return image.astype(np.float32)
    else:
        raise ValueError("Unsupported data format {}".format(data_format))

# def get_image_id(path):
#     '''
#     Assume that the path is in the format of .../split/class_name/filename.png

#     Arg(s):
#         path : str
#             path to image

#     Returns:
#         image_id : str
#             image id in format of classname-split-filename
#     '''
#     split = os.path.basename(os.path.dirname(os.path.dirname(path)))
#     class_name = os.path.basename(os.path.dirname(path))
#     file_name = os.path.basename(path).split(".")[0]
#     image_id = "{}-{}-{}".format(class_name, split, file_name)
#     return image_id

def save_image(image, save_path):
    '''
    Given the image, save as PNG to save_path

    Arg(s):
        image : np.array
            image to save
        save_path : str
            location in file system to save to

    Returns:
        None
    '''
    # Create save directory if it doesn't already exist
    ensure_dir(os.path.dirname(save_path))

    # Convert to integer values
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = image * 255.0
        image = image.astype(np.uint8)
    # Transpose if in format of C x H x W to H x W x C
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    # Convert array -> image and save
    image = Image.fromarray(image)
    image.save(save_path)

# def save_torch(data, save_dir, name, overwrite=False):
#     ensure_dir(save_dir)
#     save_path = os.path.join(save_dir, '{}.pth'.format(name))
#     if not overwrite and os.path.exists(save_path):
#         print("File {} already exists and overwrite is False. Aborting".format(save_path))
#         return
#     torch.save(data, save_path)
    
    
def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


# def get_common_dir_path(paths):
#     '''
#     Get longest common directory path from all the paths
#     '''
#     common_dir_path = paths[0]
#     for path in paths:
#         while common_dir_path not in path:
#             common_dir_path = os.path.dirname(common_dir_path)

#     return common_dir_path

def ensure_files(files):
    '''
    Given a list of file paths, return paths that don't exist

    Arg(s):
        files : list[str]
            list of file paths

    Returns:
        list[str] or empty list
    '''
    non_existent_paths = []
    for file_path in files:
        if not os.path.exists(file_path):
            non_existent_paths.append(file_path)

    return non_existent_paths


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


# def inf_loop(data_loader):
#     ''' wrapper function for endless data loader. '''
#     for loader in repeat(data_loader):
#         yield from loader


def copy_file(file_path, save_dir):
    '''
    Copy the file from file_path to the directory save_dir
    Arg(s):
        file_path : str
            file to copy
        save_dir : str
            directory to save file to
    Returns : None
    '''
    # Assert files/directories exist
    assert os.path.exists(file_path)
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, os.path.basename(file_path))
    shutil.copy(file_path, save_path)

def copy_tree(dir_path, save_dir):
    '''
    Given path to a directory, copy all files/folders recursively into save_dir

    Arg(s):
        dir_path : str
            directory containing files/folders to copy
        save_dir : str
            directory to save to
    '''
    assert os.path.exists(dir_path)
    ensure_dir(os.path.dirname(save_dir))
    if os.path.exists(save_dir):
        raise ValueError("Directory {} already exists".format(save_dir))
    shutil.copytree(dir_path, save_dir)




def list_to_dict(list_, list_2=None):
    '''
    Given a list, return a dictionary keyed by the elements of the list to corresponding indices
    If list_2 is not None, return dictionary mapping list_ to list_2 objects

    Arg(s):
        list_ : list[any]
            input list
        list_2 : list[any] or None
            must be same length as list_

    Returns:
        dict_: dict{ any : int}
            corresponding dictionary to list_
    '''
    dict_ = {}
    if list_2 is None:
        for idx, element in enumerate(list_):
            dict_[element] = idx
    else:
        assert len(list_) == len(list_2)
        for l1_element, l2_element in zip(list_, list_2):
            dict_[l1_element] = l2_element

    return dict_

def print_dict(dictionary, indent_level=0):
    tabs = ""
    for i in range(indent_level):
        tabs += "\t"
    for key, val in dictionary.items():
        if type(val) == dict:
            print("{}{}".format(tabs, key))
            print_dict(val, indent_level=indent_level+1)
        else:

            print("{}{} : {}".format(tabs, key, val))

def make_log_path(save_dir,
                  log_id=None):
    '''
    Given the save directory, return path to log file
    '''
    log_dir = os.path.join(save_dir, 'logs')
    if log_id is not None:
        log_path = os.path.join(log_dir, '{}_{}.txt'.format(log_id, datetime.now().strftime('%m%d%Y_%H%M%S')))
    else:
        log_path = os.path.join(log_dir, '{}.txt'.format(datetime.now().strftime('%m%d%Y_%H%M%S')))
    ensure_dir(log_dir)
    return log_path

def informal_log(s, filepath=None, to_console=True, timestamp=True):
    '''
    Logs a string to either file or console
    Arg(s):
        s : str
            string to log
        filepath
            output filepath for logging
        to_console : bool
            log to console
    '''
    if timestamp:
        s = '[{}] {}'.format(datetime.now().strftime(r'%m%d_%H%M%S'), s)
    if to_console:
        print(s)

    if filepath is not None:
        if len(os.path.dirname(filepath)) > 0 and not os.path.isdir(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            with open(filepath, 'w+') as o:
               o.write(s + '\n')
        else:
            with open(filepath, 'a+') as o:
                o.write(s + '\n')

def run_bash(command):
    # Run subprocess command
    subprocess.run(command, shell=True, executable='/bin/bash')
    try:
        result = subprocess.check_output(command, shell = True, executable = "/bin/bash", stderr = subprocess.STDOUT)
    except subprocess.CalledProcessError as cpe:
        result = cpe.output
        raise ValueError("Subprocess call failed: {}".format(result))

class DictObj:
    def __init__(self, in_dict):
        for key, val in in_dict.items():
            if key.startswith("_"):
                continue
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)