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
from .. import kagglebirds2020


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
            if 'label_mask' in annotations:
                shapes['label_mask'] = ()
                dtypes['label_mask'] = np.float32

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
        'label_mask': list(data['Y_mask'])
    }
    train_csv = pd.DataFrame(d, index=d['sample_key'])
    annotations = pd.DataFrame(
        {'label_all': d['label_all'], 'label_mask': d['label_mask']}, index=d['sample_key'])
    ignore = ~np.stack(annotations.label_mask)
    labels = np.stack(annotations.label_all)
    labels = np.round(labels)
    labels[ignore] = 255
    annotations.label_all = pd.Series(list(labels), index=annotations.index)

    with open(os.path.join(os.path.join(here, cfg['data.label-map'])), 'r') as f:
        class_map = json.load(f)

    # derive set of labels, ordered by latin names
    labelset_ebird = class_map.keys()

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
    dataset = OpenmicDataset(
        itemids, audios, labelset_ebird, annotations=annotations)

    # unified length, if needed
    if cfg['data.len_min'] < cfg['data.len_max']:
        raise NotImplementedError(
            "data.len_min < data.len_max not allowed yet")
    elif cfg['data.len_max'] > 0:
        dataset = kagglebirds2020.FixedSizeExcerpts(dataset,
                                                    int(sample_rate *
                                                        cfg['data.len_min']),
                                                    deterministic=designation != 'train')

    # convert to float and move channel dimension to the front
    dataset = kagglebirds2020.Floatify(dataset, transpose=True)

    dataset = kagglebirds2020.DownmixChannels(dataset,
                                              method=(cfg['data.downmix']
                                                      if designation == 'train'
                                                      else 'average'))

    if cfg['data.class_sample_weights'] and designation == 'train':
        class_weights = cfg['data.class_sample_weights']
        if class_weights not in ('equal', 'roundrobin'):
            class_weights = list(map(float, class_weights.split(',')))
        dataset.sampler = kagglebirds2020.ClassWeightedRandomSampler(train_csv.label_all,
                                                     class_weights)

    return dataset
