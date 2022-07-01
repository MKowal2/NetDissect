

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


def ensure_dir(targetdir):
    if not os.path.isdir(targetdir):
        try:
            os.makedirs(targetdir)
        except:
            pass

if __name__ == '__main__':
    # general imports
    import argparse
    from collections import OrderedDict
    from scipy.misc import imread, imresize, imsave
    from scipy.ndimage.interpolation import zoom
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
    categories = ['dynamics']
    dtdb = dtdb_dataset.DTDB(data_root='/home/m2kowal/data/DTDB',
                             categories=categories)

    data = OrderedDict(dtdb=dtdb, a2d=a2d)

    unify(data,
            splits=OrderedDict(train=0.7, val=0.3),
            size=image_size, segmentation_size=seg_size,
            directory=('dataset/broden1_%d' % args.size),
            synonyms=synonyms,
            min_frequency=10, min_coverage=0.5, verbose=True)