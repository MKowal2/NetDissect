from multiprocessing import Pool, cpu_count
from data_utils import *
import signal
import numpy
from functools import partial

def unify(data_sets, directory, size=None, segmentation_size=None, crop=False,
        splits=None, min_frequency=None, min_coverage=None,
        synonyms=None, test_limit=None, single_process=False, verbose=False):

    # create directory
    ensure_dir(os.path.join(directory, 'videos'))
    # Phase 1: Count label statistics
    # frequency = number of images touched by each label
    # coverage  = total portion of images covered by each label
    frequency, coverage = gather_label_statistics(
            data_sets, test_limit, single_process,
            segmentation_size, crop, verbose)
    # Phase 2: Sort, collapse, and filter labels
    labnames, syns = normalize_labels(data_sets, frequency, coverage, synonyms)
    report_aliases(directory, labnames, data_sets, frequency, verbose)
    # Phase 3: Filter by frequncy, and assign numbers
    names, assignments = assign_labels(
            labnames, frequency, coverage, min_frequency, min_coverage, verbose)
    # Phase 4: Collate and output label stats
    cats = write_label_files(
            directory, names, assignments, frequency, coverage, syns, verbose)
    # Phase 5: Create normalized segmentation files
    create_segmentations(
            directory, data_sets, splits, assignments, size, segmentation_size,
            crop, cats, test_limit, single_process, verbose)


def gather_label_statistics(data_sets, test_limit, single_process,
        segmentation_size, crop, verbose):
    '''
    Phase 1 of unification.  Counts label statistics.
    '''
    # Count frequency and coverage for each individual image
    stats = map_in_pool(partial(count_label_statistics,
        segmentation_size=segmentation_size,
        crop=crop,
        verbose=verbose),
            all_dataset_segmentations(data_sets, test_limit),
            single_process=single_process,
            verbose=verbose)
    # Add them up
    frequency, coverage = (sum_histogram(d) for d in zip(*stats))
    # TODO: also, blacklist images that have insufficient labled pixels
    return frequency, coverage

def count_label_statistics(record, segmentation_size, crop, verbose):
    '''
    Resolves the segmentation record, and then counts all nonzero
    labeled pixels in the resulting segmentation.  Returns two maps:
      freq[(dataset, category, label)] = 1, if the label is present
      coverage[(dataset, category, label)] = p, for portion of pixels covered
    '''
    dataset, index, seg_class, fn, md = record
    if verbose:
        print( 'Counting #%d %s %s' % (index, dataset, os.path.dirname(fn)))

    # get dict of all segmentation maps (e.g., flow-bins, actions, etc)
    full_seg, shape = seg_class.resolve_segmentation(md)
    freq = {}
    coverage = {}

    # TODO: implement video statistics along with image statistics

    # calculate the frequency and coverage of
    for category, seg in full_seg.items():
        if seg is None:
            continue
        dims = len(numpy.shape(seg))
        if dims <= 1:
            for label in (seg if dims else (seg,)):
                key = (dataset, category, int(label))
                freq[key] = 1
                coverage[key] = 1.0
        elif dims >= 2:
            # We do _not_ scale the segmentation for counting purposes!
            # Different scales should produce the same labels and label order.
            # seg = scale_segmentation(seg, segmentation_size, crop=crop)
            bc = numpy.bincount(seg.ravel())
            pixels = numpy.prod(seg.shape[-2:])
            for label in bc.nonzero()[0]:
                if label > 0:
                    key = (dataset, category, int(label))
                    freq[key] = 1
                    coverage[key] = float(bc[label]) / pixels
    return freq, coverage

def map_in_pool(fn, data, single_process=False, verbose=False):
    '''
    Our multiprocessing solution; wrapped to stop on ctrl-C well.
    '''
    if single_process:
        return map(fn, data)
    n_procs = min(cpu_count(), 12)
    original_sigint_handler = setup_sigint()
    pool = Pool(processes=n_procs, initializer=setup_sigint)
    restore_sigint(original_sigint_handler)
    try:
        if verbose:
            print( 'Mapping with %d processes' % n_procs)
        res = pool.map_async(fn, data)
        return res.get(31536000)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        raise
    else:
        pool.close()
        pool.join()

def sum_histogram(histogram_list):
    '''Adds histogram dictionaries elementwise.'''
    result = {}
    for d in histogram_list:
        for k, v in d.iteritems():
            if k not in result:
                result[k] = v
            else:
                result[k] += v
    return result

def setup_sigint():
    return signal.signal(signal.SIGINT, signal.SIG_IGN)

def restore_sigint(original):
    signal.signal(signal.SIGINT, original)

def all_dataset_segmentations(data_sets, test_limit=None):
    '''
    Returns an iterator for metadata over all segmentations
    for all images in all the datasets.  The iterator iterates over
    (dataset_name, global_index, dataset_resolver, metadata(i))
    '''
    j = 0
    for name, ds in data_sets.items():
        for i in truncate_range(range(ds.size()), test_limit):
            yield (name, j, ds.__class__, ds.filename(i), ds.metadata(i))
            j += 1

def truncate_range(data, limit):
    '''For testing, if limit is not None, limits data by slicing it.'''
    if limit is None:
        return data
    if isinstance(limit, slice):
        return data[limit]
    return data[:limit]

if __name__ == '__main__':
    # general imports
    import argparse
    from collections import OrderedDict
    import shutil
    import sys
    import os

    # dataset imports
    # TODO: build dataloaders for each dataset individually
    import dtdb_dataset
    import a2d_dataset


    parser = argparse.ArgumentParser(
        description='Generate video broden dataset.')
    parser.add_argument(
            '--size',
            type=int, default=224,
            help='pixel size for input videos')
    args = parser.parse_args()

    image_size = (args.size, args.size)
    seg_size = (args.size // 2, args.size // 2)

    print('CREATING NEW DATASET OF SIZE {}'.format(image_size))
    print('CREATING NEW LABELS OF SIZE {}'.format(seg_size))
    print('Loading source segmentations.')
    # categories = ['dynamics', 'data_processing']
    categories = ['dynamics', 'appearance', 'color', 'flow']
    dtdb = dtdb_dataset.DTDB(data_root='/home/m2kowal/data/DTDB',
                             categories=categories,
                             min_video_frame_length = 63)

    data = OrderedDict(dtdb=dtdb)

    unify(data,
            splits=OrderedDict(train=0.7, val=0.3),
            size=image_size, segmentation_size=seg_size,
            directory=('dataset/broden1_%d' % args.size),
            synonyms=None,
            min_frequency=10, min_coverage=0.5, single_process=False,verbose=True)