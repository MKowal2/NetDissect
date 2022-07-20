from data_utils import *
import numpy
from functools import partial
import codecs
import csv
from PIL import Image



def unify(data_sets, directory, size=None, segmentation_size=None, crop=False,
        splits=None, min_frequency=None, min_coverage=None, min_vid_frequency=None,
        synonyms=None, test_limit=None, single_process=False, verbose=False, debug=False, num_threads=1):


    # create directory
    ensure_dir(directory)
    ensure_dir(os.path.join(directory, 'images'))

    write_readme_file([
        ('data_sets', data_sets), ('size', size),
        ('segmentation_size', segmentation_size), ('crop', crop),
        ('splits', splits),
        ('min_frequency', min_frequency), ('min_coverage', min_coverage),
        ('video_min_frequency', min_vid_frequency),
        ('synonyms', synonyms), ('test_limit', test_limit),
        ('single_process', single_process)],
        directory=directory, verbose=verbose)


    # Phase 1: Count label statistics
    # frequency = number of images touched by each label
    # coverage  = total portion of images covered by each label
    frequency, coverage, video_frequency = gather_label_statistics(
            data_sets, test_limit, single_process,
            segmentation_size, crop, verbose, debug,num_threads)
    # Phase 2: Sort, collapse, and filter labels
    labnames, syns = normalize_labels(data_sets, frequency, coverage, synonyms)
    report_aliases(directory, labnames, data_sets, frequency, verbose)
    # Phase 3: Filter by frequncy, and assign numbers
    names, assignments = assign_labels(
            labnames, frequency, coverage, video_frequency, min_frequency, min_coverage, min_vid_frequency, verbose)
    # Phase 4: Collate and output label stats
    cats = write_label_files(
            directory, names, assignments, frequency, coverage, video_frequency, syns, verbose)
    # Phase 5: Create normalized segmentation files
    create_segmentations(
            directory, data_sets, splits, assignments, size, segmentation_size,
            crop, cats, test_limit, single_process, verbose)


def gather_label_statistics(data_sets, test_limit, single_process,
        segmentation_size, crop, verbose, debug, num_threads):
    '''
    Phase 1 of unification.  Counts label statistics.
    '''
    # Count frequency and coverage for each individual image
    video_stats = {}
    # img_stats, video_stats = map_in_pool(partial(count_label_statistics,
    stats = map_in_pool(partial(count_label_statistics,
        segmentation_size=segmentation_size,
        crop=crop,
        verbose=verbose,
        video_stats=video_stats),
            all_dataset_segmentations(data_sets, test_limit, debug=debug),
            single_process=single_process,
            verbose=verbose,
                        num_threads=num_threads)

    # Add them up
    for idx, d in enumerate(zip(*stats)):
        if idx == 0:
            # calculate image frequency
            frequency = sum_histogram(d)
            # calculate video frequency
            video_frequency = video_sum_histogram(d)
        elif idx == 1:
            # image and video coverage are the same since it considers every pixel in the dataset independantly
            coverage = sum_histogram(d)
    # frequency, coverage = (sum_histogram(d) for d in zip(*stats))
    return frequency, coverage, video_frequency

def count_label_statistics(record, segmentation_size, crop, verbose, video_stats):
    '''
    Resolves the segmentation record, and then counts all nonzero
    labeled pixels in the resulting segmentation.  Returns two maps:
      freq[(dataset, category, label)] = 1, if the label is present
      coverage[(dataset, category, label)] = p, for portion of pixels covered
    '''
    dataset, index, seg_class, fn, md = record
    video_name = md[-1]
    if verbose:
        print( 'Counting #%d %s %s' % (index, dataset, os.path.dirname(fn)))

    # if video_name not in video_stats.keys():
        # print(1)

    # get dict of all segmentation maps (e.g., flow-bins, actions, etc)
    full_seg, shape = seg_class.resolve_segmentation(md)
    freq = {}
    coverage = {}

    # calculate the frequency and coverage of
    for category, seg in full_seg.items():
        if seg is None:
            continue
        dims = len(numpy.shape(seg))
        freq['video_name'] = video_name
        coverage['video_name'] = video_name
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


def normalize_labels(data_sets, frequency, coverage, synonyms):
    '''
    Phase 2 of unification.
    Assigns unique label names and resolves duplicates by name.
    '''
    # Sort by frequency, descending, and assign a name to each label
    top_syns = {}
    labname = {}
    freq_items = list(zip(*sorted((-f, lab) for lab, f in frequency.items())))[1]
    for lab in freq_items:
        dataset, category, label = lab
        names = [n.lower() for n in data_sets[dataset].all_names(
              category, label) if len(n) and n != '-']
        if synonyms:
            names = synonyms(names)
        # Claim the synonyms that have not already been taken
        for name in names:
            if name not in top_syns:
                top_syns[name] = lab
        # best_labname may decide to collapse two labels because they
        # have the same names and seem to mean the same thing
        labname[lab], unique = best_labname(
                lab, names, labname, top_syns, coverage, frequency)
    return labname, top_syns

def report_aliases(directory, labnames, data_sets, frequency, verbose):
    '''
    Phase 2.5
    Report the aliases.  These are printed into 'syns.txt'
    '''
    show_all = True  # Print(all details to help debugging)
    name_to_lab = invert_dict(labnames)
    with codecs.open(os.path.join(directory, 'syns.txt'), 'w', 'utf-8') as f:
        def report(txt):
            f.write('%s\n' % txt)
            if verbose:
                print(txt)
        for name in sorted(name_to_lab.keys(),
                key=lambda n: (-len(name_to_lab[n]), n)):
            keys = name_to_lab[name]
            if not show_all and len(keys) <= 1:
                break
            # Don't bother reporting aliases if all use the same index;
            # that is probably due to an underlying shared code.
            if not show_all and len(set(i for d, c, i in keys)) <= 1:
                continue
            report('name: %s' % name)
            for ds, cat, i in keys:
                names = ';'.join(data_sets[ds].all_names(cat, i))
                freq = frequency[(ds, cat ,i)]
                report('%s/%s#%d: %d, (%s)' % (ds, cat, i, freq, names))
            report('')

def assign_labels(
        labnames, frequency, coverage, video_frequency, min_frequency, min_coverage, min_vid_frequency, verbose):
    '''
    Phase 3 of unification.
    Filter names that are too infrequent, then assign numbers.
    '''
    # Collect by-name frequency and coverage
    name_frequency = join_histogram(frequency, labnames)
    video_name_frequency = join_histogram(video_frequency, labnames)
    name_coverage = join_histogram(coverage, labnames)
    names = name_frequency.keys()

    # filter by image frquency, coverage, video frequency
    if min_frequency is not None:
        names = [n for n in names if name_frequency[n] >= min_frequency]
    if min_vid_frequency is not None:
        names = [n for n in names if video_name_frequency[n] >= min_frequency]
    if min_coverage is not None:
        names = [n for n in names if name_coverage[n] >= min_coverage]
    # Put '-' at zero
    names = [n for n in names if n != '-']
    names = ['-'] + sorted(names,
            key=lambda x: (-name_frequency[x], -name_coverage[x]))
    nums = dict((n, i) for i, n in enumerate(names))
    assignments = dict((k, nums.get(v, 0)) for k, v in labnames.items())
    return names, assignments

def write_label_files(
        directory, names, assignments, frequency, coverage, video_frequency, syns, verbose):
    '''
    Phase 4 of unification.
    Collate some stats and then write then to two metadata files.
    '''
    # Make lists of synonyms claimed by each label
    synmap = invert_dict(dict((w, assignments[lab]) for w, lab in syns.items()))
    # We need an (index, category) count
    ic_freq = join_histogram_fn(frequency, lambda x: (assignments[x], x[1]))
    vid_ic_freq = join_histogram_fn(video_frequency, lambda x: (assignments[x], x[1]))
    ic_cov = join_histogram_fn(coverage, lambda x: (assignments[x], x[1]))
    for z in [(j, cat) for j, cat in ic_freq if j == 0]:
        del ic_freq[z]
        del vid_ic_freq[z]
        del ic_cov[z]
    catstats = [[] for n in names]
    vid_catstats = [[] for n in names]
    # For each index, get a (category, frequency) list in descending order
    for (ind, cat), f in sorted(ic_freq.items(), key=lambda x: -x[1]):
        catstats[ind].append((cat, f))
    for (ind, cat), f in sorted(vid_ic_freq.items(), key=lambda x: -x[1]):
        vid_catstats[ind].append((cat, f))
    index_coverage = join_histogram(coverage, assignments)
    with open(os.path.join(directory, 'label.csv'), 'w') as csvfile:
        fields = ['number', 'name', 'category', 'frequency', 'coverage', 'vid_frequency', 'syns']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for ind, name in enumerate(names):
            if ind == 0:
                continue
            writer.writerow(dict(
                number='%d' % ind,
                name=name,
                category=';'.join('%s(%d)' % s for s in catstats[ind]),
                frequency='%d' % sum(f for c, f in catstats[ind]),
                coverage='%f' % index_coverage[ind],
                vid_frequency='%d' % sum(f for c, f in vid_catstats[ind]),
                syns=';'.join([s for s in synmap[ind] if s != name])
            ))
    # For each category, figure the first, last, and other stats
    cat_ind = [(cat, ind) for ind, cat in ic_freq.keys()]
    first_index = build_histogram(cat_ind, min)
    last_index = build_histogram(cat_ind, max)
    count_labels = build_histogram([(cat, 1) for cat, _ in cat_ind])
    cat_freq = join_histogram_fn(ic_freq, lambda x: x[1])
    vid_cat_freq = join_histogram_fn(vid_ic_freq, lambda x: x[1])
    cats = sorted(first_index.keys(), key=lambda x: first_index[x])
    with open(os.path.join(directory, 'category.csv'), 'w') as csvfile:
        fields = ['name', 'first', 'last', 'count', 'frequency', 'vid_frequency']
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for cat in cats:
            writer.writerow(dict(
                name=cat,
                first=first_index[cat],
                last=last_index[cat],
                count=count_labels[cat],
                frequency=cat_freq[cat],
                vid_frequency=vid_cat_freq[cat]))
    # And for each category, create a dense coding file.
    for cat in cats:
        dense_code = [0] + sorted([i for i, c in ic_freq if c == cat],
                key=lambda i: (-ic_freq[(i, cat)], -ic_cov[(i, cat)]))
        fields = ['code', 'number', 'name', 'frequency', 'coverage', 'vid_frequency']
        with open(os.path.join(directory, 'c_%s.csv' % cat), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            for code, i in enumerate(dense_code):
                if code == 0:
                    continue
                writer.writerow(dict(
                    code=code,
                    number=i,
                    name=names[i],
                    frequency=ic_freq[(i, cat)],
                    coverage=ic_cov[(i, cat)],
                    vid_frequency=vid_ic_freq[(i, cat)]))
    return cats



def create_segmentations(directory, data_sets, splits, assignments, size,
        segmentation_size, crop, cats, test_limit, single_process, verbose):
    '''
    Phase 5 of unification.  Create the normalized segmentation files
    '''
    if size is not None and segmentation_size is None:
        segmentation_size = size
    # Get assignments into a nice form, once, here.
    # (dataset, category): [numpy array with new indexes]
    index_max = build_histogram(
            [((ds, cat), i) for ds, cat, i in assignments.keys()], max)
    index_mapping = dict([k, numpy.zeros(i + 1, dtype=numpy.int16)]
            for k, i in index_max.items())
    for (ds, cat, oldindex), newindex in assignments.items():
        index_mapping[(ds, cat)][oldindex] = newindex
    # Count frequency and coverage for each individual image
    segmented = map_in_pool(
            partial(translate_segmentation,
                directory=directory,
                mapping=index_mapping,
                size=size,
                segmentation_size=segmentation_size,
                categories=cats,
                crop=crop,
                verbose=verbose),
            all_dataset_segmentations(data_sets, test_limit),
            single_process=single_process,
            verbose=verbose)
    # Sort nonempty items randomly+reproducibly by md5 hash of the filename.
    ordered = sorted([(hashed_float(r['image']), r) for r in segmented if r])
    # Assign splits, pullout out last 20% for validation.
    cutoffs = cumulative_splits(splits)
    for floathash, record in ordered:
        for name, cutoff in cutoffs:
            if floathash <= cutoff:
                record['split'] = name
                break
        else:
            assert False, 'hash %f exceeds last split %f' % (floathash, c)

    # Now write one row per image and one column per category
    with open(os.path.join(directory, 'index.csv'), 'w') as csvfile:
        fields = ['image', 'split', 'ih', 'iw', 'sh', 'sw'] + cats
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for f, record in ordered:
            writer.writerow(record)

def translate_segmentation(record, directory, mapping, size,
        segmentation_size, categories, crop, verbose):
    '''
    Translates a single segmentation.
    '''
    dataset, index, seg_class, filename, md = record
    file_dirs = filename.split('/')
    video_name, basename = file_dirs[-2], file_dirs[-1]
    if verbose:
        print( 'Processing #%d %s %s' % (index, dataset, filename))
    full_seg, shape = seg_class.resolve_segmentation(md)
    # Rows can be omitted by returning no segmentations.
    if not full_seg:
        return None
    jpg = numpy.asarray(Image.open(filename))
    if size is not None:
        jpg = scale_image(jpg, size, crop)
        for cat in full_seg:
            full_seg[cat] = scale_segmentation(
                    full_seg[cat], segmentation_size, crop)
    else:
        size = jpg.shape[:2]
        segmentation_size = (1,1)
        for cat in full_seg:
            if len(numpy.shape(full_seg[cat])) >= 2:
                segmentation_size = numpy.shape(full_seg[cat])
                break

    imagedir = os.path.join(directory, 'images')
    ensure_dir(os.path.join(imagedir, dataset))
    ensure_dir(os.path.join(imagedir, dataset, video_name))
    fn = save_image(jpg, imagedir, dataset, os.path.join(video_name,basename))
    result = {
            'image': os.path.join(dataset, fn),
            'ih': size[0],
            'iw': size[1],
            'sh': segmentation_size[0],
            'sw': segmentation_size[1]
    }
    for cat in full_seg:
        if cat not in categories:
            continue  # skip categories with no data globally
        result[cat] = ';'.join(save_segmentation(full_seg[cat],
                imagedir, dataset, fn, cat, mapping[(dataset, cat)]))
    return result


def cumulative_splits(splits):
    '''Converts [0.8, 0.1, 0.1] to [0.8, 0.9, 1.0]'''
    if splits is None:
        return [('train', 1.0)]  # Default to just one split.
    result = []
    c = 0.0
    for name, s in splits.items():
        c += s
        result.append((name, c))
    # Eliminate any fp rounding problem: round last split up to 1.0
    if result[-1][1] < 1.0 - len(result) * sys.float_info.epsilon:
        raise ValueError('splits must add to 1.0, but %s add to %s' % (
            repr(splits), result[-1][1]))
    result[-1] = (result[-1][0], 1.0)
    return result


if __name__ == '__main__':
    # general imports
    import argparse
    from collections import OrderedDict
    import shutil
    import sys
    import os

    # dataset imports
    import dtdb_dataset
    import a2d_dataset


    parser = argparse.ArgumentParser(
        description='Generate video broden dataset.')
    parser.add_argument(
            '--size',
            type=int, default=224,
            help='pixel size for input videos')
    parser.add_argument(
            '--min_frequency',
            type=int, default=1,
            help='minimum number of images touched by each label to keep label')
    parser.add_argument(
            '--min_coverage',
            type=int, default=1,
            help='total portion of pixels touched on an per-image bases')
    parser.add_argument(
            '--min_vid_frequency',
            type=int, default=1,
            help='minimum number of videos touched by each label to keep label')
    parser.add_argument(
            '--num_threads',
            type=int, default=8,
            help='number of threads')
    parser.add_argument(
            '--single_proc',
            action='store_true',
            help='Whether to use multi_proc')
    parser.add_argument(
            '--debug',
            action='store_true',
            help='Whether to debug and only use a sample of data')
    args = parser.parse_args()

    image_size = (args.size, args.size)
    seg_size = (args.size // 2, args.size // 2)

    print('CREATING NEW DATASET OF SIZE {}'.format(image_size))
    print('CREATING NEW LABELS OF SIZE {}'.format(seg_size))
    print('Loading source segmentations.')
    # categories = ['dynamics', 'data_processing']
    categories = ['dynamics', 'appearance', 'color', 'flow']
    # categories = ['color', 'flow']
    dtdb = dtdb_dataset.DTDB(data_root='/home/m2kowal/data/DTDB',
                             categories=categories,
                             min_video_frame_length=128,
                             center_crop=128)

    a2d = a2d_dataset.A2D(data_root='/home/m2kowal/data/a2d_dataset',
                             categories=categories,
                             min_video_frame_length=64,
                             choose_ann_idx=0) # choose_ann_idx is the idx of the annotations to choos




    data = OrderedDict(dtdb=dtdb, a2d=a2d)
    # data = OrderedDict(a2d=a2d)


    # Debug settings
    min_frequency = args.min_frequency
    min_coverage = args.min_coverage
    min_vid_frequency = args.min_vid_frequency
    single_process = args.single_proc
    debug = args.debug
    num_threads = args.num_threads



    # single_process = True
    # debug = True

    directory = 'dataset/video_broden3_%d' % args.size
    unify(data,
            splits=OrderedDict(train=0.7, val=0.3),
            size=image_size, segmentation_size=seg_size,
            directory=directory,
            synonyms=None,
            min_frequency=min_frequency, # needs to be in at least 1000 frames
            min_coverage=min_coverage, # total sum of pixel labels needs to be over 50 frames
            min_vid_frequency=min_vid_frequency, # needs to be in at least 10 different videos
            single_process=single_process,
            verbose=True,
            debug=debug,
            num_threads=num_threads)