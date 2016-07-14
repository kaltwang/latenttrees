import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.backends
import os.path

from misc.numpy_helper import normalize_convex

# optional imports:
try:
    import pyperclip
except ImportError:
    pass
try:
    import textwrap
except ImportError:
    pass
try:
    import pylab
except ImportError:
    pass

def is_message_1to2(id_node1, id_node2, id_dest):
        """returns the type of message and the source id.
        :param id_node1: edge first node
        :param id_node2: edge second node
        :param id_dest: message destination
        :return: (is_message_1to2, id_src)
        """
        if id_node1 == id_dest:
            # message is reverse of edge direction -> beta
            is_message_1to2 = False
            id_src = id_node2
        else:
            # message is in edge direction -> alpha
            assert id_node2 == id_dest
            is_message_1to2 = True
            id_src = id_node1
        return is_message_1to2, id_src

def logdot(a, b):
    # from
    # http://stackoverflow.com/questions/23630277/numerically-stable-way-to-multiply-log-probability-matrices-in-numpy
    max_a, max_b = np.max(a), np.max(b)
    exp_a, exp_b = a - max_a, b - max_b
    np.exp(exp_a, out=exp_a)
    np.exp(exp_b, out=exp_b)
    c = np.dot(exp_a, exp_b)
    np.log(c, out=c)
    c += max_a + max_b
    return c

def imshow_values(data, ax=None, show_value_text=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8*2, 6*2))

    min_val_x, max_val_x, diff_x = 0, data.shape[0], 1
    min_val_y, max_val_y, diff_y = 0, data.shape[1], 1

    data_max = np.nanmax(data)
    data_min = np.nanmin(data)
    if np.isnan(data_max) or np.isnan(data_min):
        data_max = 0
        data_min = 0
    c_max = np.maximum(np.abs(data_max), np.abs(data_min))

    #imshow portion
    ax.imshow(data, interpolation='nearest', cmap='bwr', vmin = -c_max, vmax=c_max)

    if show_value_text:
        #text portion
        ind_array_x = np.arange(min_val_x, max_val_x, diff_x)
        ind_array_y = np.arange(min_val_y, max_val_y, diff_y)
        x, y = np.meshgrid(ind_array_x, ind_array_y)

        for x_val, y_val in zip(x.flatten(), y.flatten()):
            data_curr = data[x_val, y_val]
            c = "{:1.3f}".format(data_curr)
            if data_curr > 0:
                weight = 'bold'
            else:
                weight = 'normal'
            ax.text(y_val, x_val, c, va='center', ha='center', weight=weight)  # x and y are switched in axis representation

        #set tick marks for grid
        ax.set_yticks(np.arange(min_val_x, max_val_x))
        ax.set_xticks(np.arange(min_val_y, max_val_y))

def calc_lklhd_parent_messages(messages_prod_parent_except_child, message_child2parent):
        messages_prod_parent_except_child.set_log_const_zero()
        messages_prod_parent_except_child.prod([message_child2parent])
        lklhd = messages_prod_parent_except_child.get_log_const()
        return lklhd

def is_backend_module(fname):
    """Identifies if a filename is a matplotlib backend module"""
    return fname.startswith('backend_') and fname.endswith('.py')

def backend_fname_formatter(fname):
    """Removes the extension of the given filename, then takes away the leading 'backend_'."""
    return os.path.splitext(fname)[0][8:]

def check_matplotlib_backends():
    # from http://stackoverflow.com/questions/5091993/list-of-all-available-matplotlib-backends
    # get the directory where the backends live
    backends_dir = os.path.dirname(matplotlib.backends.__file__)

    # filter all files in that directory to identify all files which provide a backend
    backend_fnames = filter(is_backend_module, os.listdir(backends_dir))

    backends = [backend_fname_formatter(fname) for fname in backend_fnames]

    print("supported backends: \t" + str(backends))

    # validate backends
    backends_valid = []
    for b in backends:
        try:
            plt.switch_backend(b)
            backends_valid += [b]
        except:
            continue

    print("valid backends: \t" + str(backends_valid))


    # try backends performance
    for b in backends_valid:

        pylab.ion()
        try:
            plt.switch_backend(b)


            pylab.clf()
            tstart = time.time()               # for profiling
            x = range(0,2*pylab.pi,0.01)            # x-array
            line, = pylab.plot(x,pylab.sin(x))
            for i in range(1,200):
                line.set_ydata(pylab.sin(x+i/10.0))  # update the data
                pylab.draw()                         # redraw the canvas

            print(b + ' FPS: \t' , 200/(time.time()-tstart))
            pylab.ioff()

        except:
            print(b + " error :(")


def random_cat_sample(probabilities, size=(1,)):
    # http://stackoverflow.com/questions/11373192/generating-discrete-random-variables-with-specified-weights-using-scipy-or-numpy
    bins = np.cumsum(probabilities)
    return np.digitize(np.random.random_sample(size), bins)

class NanSelect(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, weight_matrix):
        idx_good = ~np.isnan(weight_matrix)
        if np.any(idx_good):
            weight_matrix_good = weight_matrix[idx_good]

            idx_select = self.func(weight_matrix_good)

            idx_good_where = np.where(idx_good.ravel())[0]
            idx = idx_good_where[idx_select]
            id1, id2 = np.unravel_index(idx, weight_matrix.shape)
        else:
            id1, id2 = None, None
        return id1, id2

def select_max_undecorated(weight_matrix_good):
    idx_select = np.argmax(weight_matrix_good)
    return idx_select

def select_weighted_random_undecorated(weight_matrix_good):
    probabilities = weight_matrix_good - np.min(weight_matrix_good)
    if np.allclose(probabilities, 0):
        # in case that all weights are the same, we use a uniform distribution
        probabilities[:] = 1
    normalize_convex(probabilities, axis=0)
    idx_select = random_cat_sample(probabilities)[0]
    return idx_select

def select_random_undecorated(weight_matrix_good):
    idx_select = np.random.randint(weight_matrix_good.shape[0], size=1)[0]
    return idx_select

def select_random_metropolis_undecorated(weight_matrix_good):
    probabilities = np.minimum(np.exp(weight_matrix_good), 1)
    if np.allclose(probabilities, 0):
        # in case that all weights are the same, we use a uniform distribution
        probabilities[:] = 1
    normalize_convex(probabilities, axis=0)
    idx_select = random_cat_sample(probabilities)[0]
    return idx_select

def execfile(file):
    with open(file) as f:
        code = compile(f.read(), file, 'exec')
        exec(code)

def print_for_excel(array):
    # inspired by http://stackoverflow.com/questions/22488566/how-to-paste-a-numpy-array-to-excel
    """
    Copies an array into a string format acceptable by Excel.
    Columns separated by \t, rows separated by \n
    """

    ndim = array.ndim
    if ndim == 0:
        array = array.reshape((1,1))
    elif ndim == 1:
        array = array.reshape((-1, 1))
    elif ndim > 2:
        raise ValueError('Array has ndim={}, but ndim <= 2 is required to fit into a table!'.format(ndim))

    # Create string from array
    line_strings = []
    for line in array:
        line_strings.append("\t".join(line.astype(str)).replace("\n",""))
    array_string = "\r\n".join(line_strings)

    print(array_string)
    pyperclip.copy(array_string)
    # Put string into clipboard (open, clear, set, close)
    # clipboard.OpenClipboard()
    # clipboard.EmptyClipboard()
    # clipboard.SetClipboardText(array_string)
    # clipboard.CloseClipboard()


def vars_recursive(obj, rec=1, indent=0, last_desc=''):
    succ = False
    if rec > 0:
        try:
            iter = sorted(obj.items())
            succ = True
        except AttributeError:
            try:
                iter = sorted(vars(obj).items())
                succ = True
            except TypeError:
                try:
                    iter = enumerate(obj)
                    succ = True
                except TypeError:
                    pass

    if succ:
        message = str(type(obj))
    else:
        message = str(obj)
    print_indent(last_desc, message, indent=indent)

    if succ:
        for name, attr in  iter:
            #__print_indent('{}: '.format(name), indent=indent+1, end='')
            next_desc = '{}: '.format(name)
            vars_recursive(attr, rec=rec-1, indent=indent+1, last_desc=next_desc)

def print_indent(desc, message, indent, end='\n'):
    string = str_indent(desc, message, indent, end)
    print(string, end='')

def str_indent(desc, message, indent, end='\n'):
    prefix = ''
    if indent > 1:
        prefix += '|   ' * (indent-1)
    if indent > 0:
        prefix += '+---'
    #print(message, end=end)
    width = 100
    initial_indent = prefix + desc
    subsequent_indent = '|   ' * indent + ' ' * len(desc)
    wrapper = textwrap.TextWrapper(initial_indent=initial_indent, width=width, subsequent_indent=subsequent_indent)
    return wrapper.fill(message) + end