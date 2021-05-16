# -*- coding: utf-8 -*-

"""
Openmic-2018 dataset

Author: Jakob Wögerbauer based on Birdcall dataset of Jan Schlüter
"""

import json
import os
import re
import glob
import itertools

import numpy as np
import pandas as pd
import tqdm

from ... import config
from .. import Dataset, ClassWeightedRandomSampler
from .. import audio
from .. import splitting


def common_shape(arrays):
    """
    Infers the common shape of an iterable of array-likes (assuming all are of
    the same dimensionality). Inconsistent dimensions are replaced with `None`.
    """
    arrays = iter(arrays)
    shape = next(arrays).shape
    for array in arrays:
        shape = tuple(a if a == b else None
                      for a, b in zip(shape, array.shape))
    return shape


class OpenmicDataset(Dataset):
    def __init__(self, itemids, wavs, labelset, annotations=None):
        shapes = dict(input=common_shape(wavs), itemid=())
        dtypes = dict(input=wavs[0].dtype, itemid=str)
        num_classes = len(labelset)

        if annotations is not None:
            if 'label_all' in annotations:
                shapes['label_all'] = (num_classes,)
                dtypes['label_all'] = np.float32
            if 'rating' in annotations:
                shapes['rating'] = ()
                dtypes['rating'] = np.float32

        super(OpenmicDataset, self).__init__(
            shapes=shapes,
            dtypes=dtypes,
            num_classes=num_classes,
            num_items=len(itemids),
        )
        self.itemids = itemids
        self.wavs = wavs
        self.labelset = labelset
        self.annotations = annotations

    def __getitem__(self, idx):
        # get audio
        item = dict(itemid=self.itemids[idx], input=self.wavs[idx])
        # get targets, if any
        for key in self.shapes:
            if key not in item:
                item[key] = self.annotations[key][idx]
        # return
        return item


class DownmixChannels(Dataset):
    """
    Dataset wrapper that downmixes multichannel audio to mono, either
    deterministically (method='average') or randomly (method='random_uniform').
    """

    def __init__(self, dataset, key='input', axis=0, method='average'):
        shapes = dict(dataset.shapes)
        shape = list(shapes[key])
        shape[axis] = 1
        shapes[key] = tuple(shape)
        super(DownmixChannels, self).__init__(
            shapes=shapes, dtypes=dataset.dtypes,
            num_classes=dataset.num_classes, num_items=len(dataset))
        self.dataset = dataset
        self.key = key
        self.axis = axis
        self.method = method

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        wav = item[self.key]
        num_channels = wav.shape[self.axis]
        if num_channels > 1:
            if self.method == 'average':
                wav = np.mean(wav, axis=self.axis, keepdims=True)
            elif self.method == 'random_uniform':
                weights = np.random.dirichlet(np.ones(num_channels))
                weights = weights.astype(wav.dtype)
                if self.axis == -1 or self.axis == len(wav.shape) - 1:
                    wav = np.dot(wav, weights)[..., np.newaxis]
                else:
                    weights = weights.reshape(weights.shape +
                                              (1,) *
                                              (len(wav.shape[self.axis:]) - 1))
                    wav = (wav * weights).sum(self.axis, keepdims=True)
        item[self.key] = wav
        return item


def loop(array, length):
    """
    Loops a given `array` along its first axis to reach a length of `length`.
    """
    if len(array) < length:
        array = np.asanyarray(array)
        if len(array) == 0:
            return np.zeros((length,) + array.shape[1:], dtype=array.dtype)
        factor = length // len(array)
        if factor > 1:
            array = np.tile(array, (factor,) + (1,) * (array.ndim - 1))
        missing = length - len(array)
        if missing:
            array = np.concatenate((array, array[:missing:]))
    return array


def crop(array, length, deterministic=False):
    """
    Crops a random excerpt of `length` along the first axis of `array`. If
    `deterministic`, perform a center crop instead.
    """
    if len(array) > length:
        if not deterministic:
            pos = np.random.randint(len(array) - length + 1)
            array = array[pos:pos + length:]
        else:
            l = len(array)
            array = array[(l - length) // 2:(l + length) // 2]
    return array


class FixedSizeExcerpts(Dataset):
    """
    Dataset wrapper that returns batches of random excerpts of the same length,
    cropping or looping inputs along the first axis as needed. If
    `deterministic`, will always do a center crop for too long inputs.
    """
    def __init__(self, dataset, length, deterministic=False, key='input'):
        shapes = dict(dataset.shapes)
        shapes[key] = (length,) + shapes[key][1:]
        super(FixedSizeExcerpts, self).__init__(
                shapes=shapes, dtypes=dataset.dtypes,
                num_classes=dataset.num_classes, num_items=len(dataset))
        self.dataset = dataset
        self.length = length
        self.deterministic = deterministic
        self.key = key

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        data = item[self.key]
        if len(data) < self.length:
            data = loop(data, self.length)
        elif len(data) > self.length:
            data = crop(data, self.length, deterministic=self.deterministic)
        item[self.key] = data
        return item


class Floatify(Dataset):
    """
    Dataset wrapper that converts audio samples to float32 with proper scaling,
    possibly transposing the data on the way to swap time and channels.
    """

    def __init__(self, dataset, transpose=False, key='input'):
        dtypes = dict(dataset.dtypes)
        dtypes[key] = np.float32
        shapes = dict(dataset.shapes)
        if transpose:
            shapes[key] = shapes[key][::-1]
        super(Floatify, self).__init__(
            shapes=shapes, dtypes=dtypes,
            num_classes=dataset.num_classes, num_items=len(dataset))
        self.dataset = dataset
        self.transpose = transpose
        self.key = key

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        data = item[self.key]
        if self.transpose:
            data = np.asanyarray(data).T
        item[self.key] = audio.to_float(data)
        return item


def find_files(basedir, regexp):
    """
    Finds all files below `basedir` that match `regexp`, sorted alphabetically.
    """
    regexp = re.compile(regexp)
    return sorted(fn for fn in glob.glob(os.path.join(basedir, '**'),
                                         recursive=True)
                  if regexp.match(fn))


def derive_labelset(train_csv):
    """
    Returns the set of used ebird codes, sorted by latin names.
    """
    labelset_latin = sorted(set(train_csv.primary_label))
    latin_to_ebird = dict(zip(train_csv.primary_label, train_csv.ebird_code))
    labelset_ebird = [latin_to_ebird[latin] for latin in labelset_latin]
    if len(set(labelset_ebird)) != len(labelset_ebird):
        raise RuntimeError("Inconsistent latin names in train.csv!")
    return labelset_ebird


def get_itemid(filename):
    """
    Returns the file name without path and without file extension.
    """
    return os.path.splitext(os.path.basename(filename))[0]


def create(cfg, designation):
    config.add_defaults(cfg, pyfile=__file__)
    here = os.path.dirname(__file__)

    # browse for audio files
    basedir = os.path.join(here, cfg['data.audio_dir'])
    audio_files = find_files(basedir, cfg['data.audio_regexp'])
    if cfg['debug']:
        print("Found %d audio files in %s matching %s." %
              (len(audio_files), basedir, cfg['data.audio_regexp']))
    if not audio_files:
        raise RuntimeError("Did not find any audio files in %s matching %s." %
                           (basedir, cfg['data.audio_regexp']))

    # read official train.csv file
    data = np.load(os.path.join(
        here, cfg['data.train_csv']), allow_pickle=True)
    d = {
        'sample_key': list(data['sample_key']),
        'label_all': list(data['Y_true']),
        'y_mask': list(data['Y_mask'])
    }
    train_csv = pd.DataFrame(d, index=d['sample_key'])
    labels = pd.DataFrame({'label_all': d['label_all']}, index=d['sample_key'])

    with open(os.path.join(os.path.join(here, cfg['data.label-map'])), 'r') as f:
        class_map = json.load(f)

    # derive set of labels, ordered by latin names
    labelset_ebird = class_map.keys()
    ebird_to_idx = class_map
    num_classes = len(labelset_ebird)

    # for training and validation, read and convert all required labels
    if designation in ('train', 'valid'):
        # constrain .csv rows to selected audio files and vice versa
        csv_ids = set(train_csv.index)
        audio_ids = {get_itemid(fn): fn for fn in audio_files}
        audio_ids = {k: fn for k, fn in audio_ids.items() if k in csv_ids}
        train_csv = train_csv.loc[[i in audio_ids for i in train_csv.index]]
        train_csv['audiofile'] = [audio_ids[i] for i in train_csv.index]
        if cfg['debug']:
            print("Found %d entries matching the audio files." %
                  len(train_csv))

        # train/valid split
        if designation == 'train':
            train_idxs = pd.read_csv(os.path.join(here, 'data/split01_train.csv'),
                                     header=None, squeeze=True)
            train_idxs = set(train_idxs).intersection(set(train_csv.index))
            train_csv = train_csv.loc[train_idxs]
        elif designation == 'valid':
            valid_idxs = pd.read_csv(os.path.join(here, 'data/split01_test.csv'),
                                     header=None, squeeze=True)
            valid_idxs = set(valid_idxs).intersection(set(train_csv.index))
            train_csv = train_csv.loc[valid_idxs]
        if cfg['debug']:
            print("Kept %d items for this split." % len(train_csv))

        # update audio_files list to match train_csv
        audio_files = train_csv.audiofile
        itemids = train_csv.index
    elif designation == 'test':
        itemids = audio_files

    # prepare the audio files, assume a consistent sample rate
    if not cfg.get('data.sample_rate'):
        cfg['data.sample_rate'] = audio.get_sample_rate(audio_files[0])
    sample_rate = cfg['data.sample_rate']
    # TODO: support .mp3?
    audios = [audio.WavFile(fn, sample_rate=sample_rate)
              for fn in tqdm.tqdm(audio_files, 'Reading audio',
                                  ascii=bool(cfg['tqdm.ascii']))]
    # create the dataset
    dataset = OpenmicDataset(itemids, audios, labelset_ebird, annotations=labels)

    # unified length, if needed
    if cfg['data.len_min'] < cfg['data.len_max']:
        raise NotImplementedError("data.len_min < data.len_max not allowed yet")
    elif cfg['data.len_max'] > 0:
        dataset = FixedSizeExcerpts(dataset,
                                    int(sample_rate * cfg['data.len_min']),
                                    deterministic=designation != 'train')

    # convert to float and move channel dimension to the front
    dataset = Floatify(dataset, transpose=True)

    dataset = DownmixChannels(dataset,
                              method=(cfg['data.downmix']
                                      if designation == 'train'
                                      else 'average'))

    return dataset
