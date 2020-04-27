import os
import time
import json

class HParam(object):
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
