"""Helper functions for working with audio files in NumPy."""
"""some code borrowed from https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py"""

import numpy as np
import contextlib
import librosa
import struct
import soundfile

__all__ = [
    'pcm2float',
    'float2pcm',
    'any_to_pcm',
    'pcm_to_wav',
]

def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return  float2pcm(sig, dtype='int16').tobytes()

def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm2float(np.frombuffer(byte,dtype=np.int16), dtype='float32')

def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

def any_to_pcm(any_file,
               sample_rate=None, channel=None, out_file=None):
    # load file to float32 
    y, sr = librosa.load(any_file,sr=sample_rate, mono=False)

    if channel == 1:
        y = librosa.to_mono(y) 

    # flatten to order of "left0,right0,left1,right1" ...
    if y.shape[0] == 2:
        y = y.flatten(order="F")

    # convert to byte(PCM16)
    byt = float_to_byte(y)

    if out_file == None:
        out_file = any_file+".pcm"
    # save to pcm file
    with open(out_file,"wb") as f:
        f.write(byt)

    return out_file

def pcm_to_wav(pcm_file, sample_rate=44100, channel=2, out_file=None):
    # read pcm file
    with open(pcm_file,"rb") as f:
        byt = f.read()

    # byte(PCM16) to float32
    y = byte_to_float(byt)
    
    # flatten to order from "left0,right0,left1,right1" ...
    y = y.reshape(channel, -1, order='F')
    # soundfile only accepts (num_samples, num_channel) format
    y = y.T

    if out_file == None:
        out_file = pcm_file+".wav"
    # save float32 to PCM16 with soundfile
    soundfile.write(out_file, y, sample_rate, 'PCM_16')

    return out_file



@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Context manager for temporarily setting NumPy print options.
    See http://stackoverflow.com/a/2891805/500098
    """
    original = np.get_printoptions()
    try:
        np.set_printoptions(*args, **kwargs)
        yield
    finally:
        np.set_printoptions(**original)

if __name__ == "__main__":
    in_file = "mix_sp-44100_ch-2_hb-16_20201221115040.wav"
    out_file = "1.wav"
    pcm_file = "1.pcm"
    any_to_pcm(in_file, sample_rate=11025, channel=1, out_file=pcm_file)
    pcm_to_wav(pcm_file, sample_rate=11025, channel=1, out_file=out_file)
