import os

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
