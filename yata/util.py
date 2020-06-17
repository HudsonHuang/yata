import os
import time
import json
import numpy as np



class HParams(object):
    """
    an alternative to 
    tf.contrib.training.HParams
    or
    from tensorboard.plugins.hparams.api.HParam
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.keys = kwargs.keys()

    def to_dict(self):
        dic = {}
        for key in self.keys:
            value = self.__getattribute__(key)
            dic[key] = value
        return dic

    def to_json(self):
        string = json.dumps(self.to_dict(),indent=2)
        return string


def new_dir(*dirname):
    # Makedirs and return path.
    # Example: 
    # - new_dir("file_a")  makedir and return "./file_a"
    # - new_dir("file_a", 1)  makedir and return "./file_a/1"
    if len(dirname) == 1:
        dirname = str(dirname[0])
    else:
        dirname = list(map(str, dirname))
        dirname = os.path.join(*dirname)
    
    dirname = os.path.abspath(dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname



def get_current_date():
    strDate = time.strftime('%Y%m%d_%H%M%S',
                            time.localtime(time.time()))
    return strDate


def to_categorical(y, num_classes=None, dtype='float32'):
    """
    copy from keras.utils.to_categorical
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def batch_clean(data, split=","):
    if type(split) == str:
        return list(map(lambda x:x.replace("\n", "").split(split), data))
    else:
        return list(map(lambda x:x.replace("\n", ""), data))
    
