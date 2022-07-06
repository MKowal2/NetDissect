import os
import signal
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import operator
import numpy
import re
from PIL import Image
from scipy.ndimage.interpolation import zoom

def ensure_dir(targetdir):
    if not os.path.exists(targetdir):
        try:
            os.mkdir(targetdir)
        except:
            pass

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
        for k, v in d.items():
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
        total_imgs = ds.img_size()
        for i in truncate_range(range(ds.img_size()), test_limit):
            yield (name, j, ds.__class__, ds.filename(i), ds.metadata(i))
            j += 1
            if j % 10000 == 0:
                print('{}/{} Completed'.format(j, total_imgs))
            if j == 300:
                break

def truncate_range(data, limit):
    '''For testing, if limit is not None, limits data by slicing it.'''
    if limit is None:
        return data
    if isinstance(limit, slice):
        return data[limit]
    return data[:limit]

def invert_dict(d):
    '''Transforms {k: v} to {v: [k,k..]}'''
    result = {}
    for k, v in d.items():
        if v not in result:
            result[v] = [k]
        else:
            result[v].append(k)
    return result

def best_labname(lab, names, assignments, top_syns, coverage, frequency):
    '''
    Among the given names, chooses the best name to assign, given
    information about previous assignments, synonyms, and stats
    '''
    # TODO: create for video broden
    # Best shot: get my own name, different from other names.
    if 'dog' in names:
        print(names)
    for name in names:
        if top_syns[name] == lab:
            return name, True
    if len(names) == 0 or len(names) == '1' and names[0] in ['', '-']:
        # If there are no names, we must use '-' and map to 0.
        return '-', False
    elif len(names) == 1:
        # If we have a conflict without ambiguity, then we just merge names.
        other = top_syns[names[0]]
    else:
        # If we need to merge and we have multiple synonyms, let's score them.
        scores = []
        # Get the average size of an object of my type
        size = coverage[lab] / frequency[lab]
        for name in names:
            other = top_syns[name]
            # Compare it to the average size of objects of other types
            other_size = coverage[other] / frequency[other]
            scores.append((abs(size - other_size), -frequency[other], other))
        # Choose the synonyms that has the most similar average size.
        # (breaking ties according to maximum frequency)
        other = min(scores)[2]
    name = assignments[other]
    return name, False


def join_histogram(histogram, newkeys):
    '''Rekeys histogram according to newkeys map, summing joined buckets.'''
    result = {}
    for oldkey, newkey in newkeys.items():
        if newkey not in result:
            result[newkey] = histogram[oldkey]
        else:
            result[newkey] += histogram[oldkey]
    return result

def build_histogram(pairs, reducer=operator.add):
    '''Creates a histogram by combining a list of key-value pairs.'''
    result = {}
    for k, v in pairs:
        if k not in result:
            result[k] = v
        else:
            result[k] = reducer(result[k], v)
    return result

def join_histogram_fn(histogram, makekey):
    '''Rekeys histogram according to makekey fn, summing joined buckets.'''
    result = {}
    for oldkey, val in histogram.items():
        newkey = makekey(oldkey)
        if newkey not in result:
            result[newkey] = val
        else:
            result[newkey] += val
    return result

def hashed_float(s):
    # Inspired by http://code.activestate.com/recipes/391413/ by Ori Peleg
    '''Hashes a string to a float in the range [0, 1).'''
    import hashlib, struct
    #TODO: fix this it is broken from the change of using md5 library
    [number] = struct.unpack(">Q", hashlib.md5(s).digest()[:8])
    return number / (2.0 ** 64)  # python will constant-fold this denominator.





def scale_image(im, dims, crop=False):
    '''
    Scales or crops a photographic image using antialiasing.
    '''
    if len(im.shape) == 2:
        # Handle grayscale images by adding an RGB channel
        im = numpy.repeat(im[numpy.newaxis], 3, axis=0)
    if im.shape[0:2] != dims:
        if not crop:
            im = Image.fromarray(im).resize((dims))
        else:
            source = im.shape[0:2]
            aspect = float(dims[1]) / dims[0]
            if aspect * source[0] > source[1]:
                width = int(dims[1] / aspect)
                margin = (width - dims[0]) // 2
                im = Image.fromarray(im).resize((width, dims[1]))[margin:margin+dims[0],:,:]
            else:
                height = int(dims[0] * aspect)
                margin = (height - dims[1]) // 2
                im = Image.fromarray(im).resize((dims[0], height))[margin:margin+dims[1],:,:]
    return im

def scale_segmentation(segmentation, dims, crop=False):
    '''
    Zooms a 2d or 3d segmentation to the given dims, using nearest neighbor.
    '''
    shape = numpy.shape(segmentation)
    if len(shape) < 2 or shape[-2:] == dims:
        return segmentation
    peel = (len(shape) == 2)
    if peel:
        segmentation = segmentation[numpy.newaxis]
    levels = segmentation.shape[0]
    result = numpy.zeros((levels, ) + dims,
            dtype=segmentation.dtype)
    ratio = (1,) + tuple(res / float(orig)
            for res, orig in zip(result.shape[1:], segmentation.shape[1:]))
    if not crop:
        safezoom(segmentation, ratio, output=result, order=0)
    else:
        ratio = max(ratio[1:])
        height = int(round(dims[0] / ratio))
        hmargin = (segmentation.shape[0] - height) // 2
        width = int(round(dims[1] / ratio))
        wmargin = (segmentation.shape[1] - height) // 2
        safezoom(segmentation[:, hmargin:hmargin+height,
            wmargin:wmargin+width],
            (1, ratio, ratio), output=result, order=0)
    if peel:
        result = result[0]
    return result

def safezoom(array, ratio, output=None, order=0):
    '''Like numpy.zoom, but does not crash when the first dimension
    of the array is of size 1, as happens often with segmentations'''
    dtype = array.dtype
    if array.dtype == numpy.float16:
        array = array.astype(numpy.float32)
    if array.shape[0] == 1:
        if output is not None:
            output = output[0,...]
        print('need zoom here')
        result = zoom(array[0,...], ratio[1:],
                output=output, order=order)
        if output is None:
            output = result[numpy.newaxis]
    else:
        result = zoom(array, ratio, output=output, order=order)
        if output is None:
            output = result
    return output.astype(dtype)

def save_image(im, imagedir, dataset, filename):
    '''
    Try to pick a unique name and save the image with that name.
    This is not race-safe, so the given name should already be unique.
    '''
    # trynum = 1
    fn = filename[:-4] + '.jpg'
    # while os.path.exists(os.path.join(imagedir, dataset, fn)):
    #     trynum += 1
    #     fn = re.sub('(?:\.jpg)?$', '%d.jpg' % trynum, filename)
    im.save(os.path.join(imagedir, dataset, fn))
    return fn

def save_segmentation(seg, imagedir, dataset, filename, category, translation):
    '''
    Saves the segmentation in a file or files if it is large, recording
    the filenames.  Or serializes it as decimal numbers to record if it
    is small.  Then returns the array of strings recorded.
    '''
    if seg is None:
        return None
    shape = numpy.shape(seg)
    if len(shape) < 2:
        # Save numbers as decimal strings; and omit zero labels.
        return ['%d' % translation[t]
                for t in (seg if len(shape) else [seg]) if t]

    result = []
    for channel in ([()] if len(shape) == 2 else range(shape[0])):
        # Save bitwise channels as filenames of PNGs; and save the files.
        #todo: check if we need encodeRG here
        im = Image.fromarray(encodeRG(translation[seg[channel]]))
        # if len(shape) == 2:
        #     fn = re.sub('(?:\.jpg)?$', '_%s.png' % category, filename)
        # else:
        #     fn = re.sub('(?:\.jpg)?$', '_%s_%d.png' %
        #             (category, channel + 1), filename)
        fn = filename[:-4] + '_flow.png'
        result.append(os.path.join(dataset, fn))
        print('need imsave here!!')
        im.save(os.path.join(imagedir, dataset, fn))
    return result

def encodeRG(channel):
    '''Encodes a 16-bit per-pixel code using the red and green channels.'''
    result = numpy.zeros(channel.shape + (3,), dtype=numpy.uint8)
    result[:,:,0] = channel % 256
    result[:,:,1] = (channel // 256)
    return result
