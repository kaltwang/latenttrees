import sys
from datetime import datetime

def crossval(s, num_folds):
    sorts = sorted(s)
    folds = [ set(sorts[i::num_folds]) for i in range(num_folds)]
    return folds

def get_path_multi_os(dict_path):
    for key, value in dict_path.items():
        if sys.platform.startswith(key):
            return value
    raise EnvironmentError('Unknown platform: ' + sys.platform)

def repr_from_properties(obj, properties):
    classname = type(obj).__name__
    repr_val = [classname]
    repr_str = "{}("
    for p in properties:
        if len(repr_val) > 1:
            repr_str += ", "
        repr_val.append(repr(getattr(obj, p)))
        repr_str += p + "={}"
    repr_str += ")"
    string = repr_str.format(*repr_val)
    return string

def binarize1_5(x):
    return x > 1.5

def get_matlab_hdf5_userblock():
    # code from https://github.com/frejanordsiek/hdf5storage
    # Get the time.
    now = datetime.now()

    # Construct the leading string. The MATLAB one looks like
    #
    # s = 'MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: ' \
    #     + now.strftime('%a %b %d %H:%M:%S %Y') \
    #     + ' HDF5 schema 1.00 .'
    #
    # Platform is going to be changed to CPython version. The
    # version is just gotten from sys.version_info, which is a class
    # for Python >= 2.7, but a tuple before that.

    v = sys.version_info
    if sys.hexversion >= 0x02070000:
        v = {'major': v.major, 'minor': v.minor, 'micro': v.micro}
    else:
        v = {'major': v[0], 'minor': v[1], 'micro': v[1]}

    s = 'MATLAB 7.3 MAT-file, Platform: CPython ' \
        + '{0}.{1}.{2}'.format(v['major'], v['minor'], v['micro']) \
        + ', Created on: ' \
        + now.strftime('%a %b %d %H:%M:%S %Y') \
        + ' HDF5 schema 1.00 .'

    # Make the bytearray while padding with spaces up to 128-12
    # (the minus 12 is there since the last 12 bytes are special.

    b = bytearray(s + (128 - 12 - len(s)) * ' ', encoding='utf-8')

    # Add 8 nulls (0) and the magic number (or something) that
    # MATLAB uses. Lengths must be gone to to make sure the argument
    # to fromhex is unicode because Python 2.6 requires it.

    b.extend(bytearray.fromhex(
        b'00000000 00000000 0002494D'.decode()))
    return b

def has_elements(iterator):
    return any(True for _ in iterator)

def isequal_or_none(a, b):
    if (a is not None) and (b is not None):
        if not a == b:
            return False
    return True

def get_and_set_attr(obj, attr_dict, setval=True):
    attr_dict_old = {}
    for attr, value_new in attr_dict.items():
        attr_dict_old[attr] = getattr(obj, attr)
        if setval:
            setattr(obj, attr, value_new)
    return attr_dict_old