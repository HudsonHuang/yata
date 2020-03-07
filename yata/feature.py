
import multiprocessing as mp
import gammatone
import pysptk
from gammatone.gtgram import gtgram
from python_speech_features import logfbank
import librosa
from ahoproc_tools.interpolate import interpolation
from ahoproc_tools.io import *
from joblib import Parallel, delayed
import pickle

try:
    import kaldi_io as kio
except ImportError:
    print ('kaldi_io is optional, but required when extracting feats with kaldi')


# Borrowed from PASE: https://github.com/santi-pdp/pase
class MIChunkWav(SingleChunkWav):
    """ Max-Information chunker expects 3 input wavs,
        and extract 3 chunks: (chunk, chunk_ctxt,
        and chunk_rand). The first two correspond to same
        context, the third one is sampled from the second wav
    """

    def __call__(self, pkg):
        pkg = format_package(pkg)
        if 'raw_rand' not in pkg:
            raise ValueError('Need at least a pair of wavs to do '
                             'MI chunking! Just got single raw wav?')
        raw = pkg['raw']
        raw_rand = pkg['raw_rand']
        self.assert_format(raw)
        self.assert_format(raw_rand)
        chunk, beg_i, end_i = self.select_chunk(raw, ret_bounds=True)
        pkg['chunk'] = chunk
        pkg['chunk_beg_i'] = beg_i
        pkg['chunk_end_i'] = end_i
        #added for parallel like corpora with close and distant mics
        #we do not make asserts here for now if raw is 
        # exactly same as raw_clean, as this was up to segmentation
        # script
        #print ("Chunk size is {}".format(chunk.size()))
        #print ("Squeezed chunk size is {}".format(chunk.squeeze(0).size()))
        if 'raw_clean' in pkg and pkg['raw_clean'] is not None:
            raw_clean = pkg['raw_clean']
            pkg['cchunk'] = self.select_chunk(raw_clean, reuse_bounds=(beg_i, end_i))
            if pkg['cchunk'] is None:
                pkg['cchunk'] = pkg['chunk']
        if 'raw_ctxt' in pkg and pkg['raw_ctxt'] is not None:
            raw_ctxt = pkg['raw_ctxt']
        else:
            # if no additional chunk is given as raw_ctxt
            # the same as current raw context is taken
            # and a random window is selected within
            raw_ctxt = raw[:]
        pkg['chunk_ctxt'] = self.select_chunk(raw_ctxt)
        pkg['chunk_rand'] = self.select_chunk(raw_rand)
        if self.random_scale:
            pkg['chunk'] = norm_and_scale(pkg['chunk'])
            pkg['chunk_ctxt'] = norm_and_scale(pkg['chunk_ctxt'])
            pkg['chunk_rand'] = norm_and_scale(pkg['chunk_rand'])
            if 'cchunk' in pkg:
                pkg['cchunk'] = norm_and_scale(pkg['cchunk'])
        # specify decimated resolution to be 1 (no decimation) so far
        pkg['dec_resolution'] = 1
        return pkg


class LPS(object):

    def __init__(self, n_fft=2048, hop=160,
                 win=400, der_order=2,
                 name='lps',
                 device='cpu'):
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.name = name
        self.der_order=der_order
        self.device = device

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        max_frames = wav.size(0) // self.hop
        if cached_file is not None:
            # load pre-computed data
            X = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            X = X[:, beg_i:end_i]
            pkg['lps'] = X
        else:
            #print ('Chunks wav shape is {}'.format(wav.shape))
            wav = wav.to(self.device)
            X = torch.stft(wav, self.n_fft,
                           self.hop, self.win)
            X = torch.norm(X, 2, dim=2).cpu()[:, :max_frames]
            X = 10 * torch.log10(X ** 2 + 10e-20).cpu()
            if self.der_order > 0 :
                deltas=[X]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(X.numpy(),order=n))
                X=torch.from_numpy(np.concatenate(deltas))
     
            pkg[self.name] = X
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(n_fft={}, hop={}, win={}'.format(self.n_fft,
                                                   self.hop,
                                                   self.win)
        attrs += ', device={})'.format(self.device)
        return self.__class__.__name__ + attrs

class FBanks(object):

    def __init__(self, n_filters=40, n_fft=512, hop=160,
                 win=400, rate=16000, der_order=2,
                 name='fbank',
                 device='cpu'):
        self.n_fft = n_fft
        self.n_filters = n_filters
        self.rate = rate
        self.hop = hop
        self.name = name
        self.win = win
        self.der_order=der_order
        self.name = name

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        if torch.is_tensor(wav):
            wav = wav.data.numpy().astype(np.float32)
        max_frames = wav.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            X = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            X = X[:, beg_i:end_i]
            pkg[self.name] = X
        else:
            winlen = (float(self.win) / self.rate)
            winstep = (float(self.hop) / self.rate)
            X = logfbank(wav, self.rate, winlen, winstep,
                         self.n_filters, self.n_fft).T
            expected_frames = len(wav) // self.hop

            if self.der_order > 0 :
                deltas=[X]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(X,order=n))
                X=np.concatenate(deltas)

            fbank = torch.FloatTensor(X)
            if fbank.shape[1] < expected_frames:
                P = expected_frames - fbank.shape[1]
                # pad repeating borders
                fbank = F.pad(fbank.unsqueeze(0), (0, P), mode='replicate')
                fbank = fbank.squeeze(0)
            pkg[self.name] = fbank
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(n_fft={}, n_filters={}, ' \
                'hop={}, win={}'.format(self.n_fft,
                                        self.n_filters,
                                        self.hop,
                                        self.win)
        return self.__class__.__name__ + attrs

class Gammatone(object):

    def __init__(self, f_min=500, n_channels=40, hop=160,
                 win=400,  der_order=2, rate=16000,
                 name='gtn',
                 device='cpu'):
        self.hop = hop
        self.win = win
        self.n_channels = n_channels
        self.rate = rate
        self.f_min = f_min
        self.der_order = der_order
        self.name = name

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        if torch.is_tensor(wav):
            wav = wav.data.numpy().astype(np.float32)
        max_frames = wav.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            X = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            X = X[:, beg_i:end_i]
            pkg[self.name] = X
        else:
            windowtime = float(self.win) / self.rate
            windowhop = float(self.hop) / self.rate
            gtn = gammatone.gtgram.gtgram(wav, self.rate, 
                                          windowtime, windowhop,
                                          self.n_channels,
                                          self.f_min)
            gtn = np.log(gtn + 1e-10)
 
            if self.der_order > 0 :
                deltas=[gtn]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(gtn,order=n))
                gtn=np.concatenate(deltas)

            expected_frames = len(wav) // self.hop
            gtn = torch.FloatTensor(gtn)
            if gtn.shape[1] < expected_frames:
                P = expected_frames - gtn.shape[1]
                # pad repeating borders
                gtn = F.pad(gtn.unsqueeze(0), (0, P), mode='replicate')
                gtn = gtn.squeeze(0)
            #pkg['gtn'] = torch.FloatTensor(gtn[:, :total_frames])

            pkg[self.name] = torch.FloatTensor(gtn)
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(f_min={}, n_channels={}, ' \
                'hop={}, win={})'.format(self.f_min,
                                        self.n_channels,
                                        self.hop,
                                        self.win)
        return self.__class__.__name__ + attrs

class LPC(object):

    def __init__(self, order=25, hop=160,
                 win=320, name='lpc',
                 device='cpu'):
        self.order = order
        self.hop = hop
        self.win = win
        self.window = pysptk.hamming(win).astype(np.float32)
        self.name = name

    def frame_signal(self, signal, window):
        
        frames = []
        for beg_i in range(0, signal.shape[0], self.hop):
            frame = signal[beg_i:beg_i + self.win]
            if len(frame) < self.win:
                # pad right size with zeros
                P = self.win - len(frame)
                frame = np.concatenate((frame,
                                        np.zeros(P,)), axis=0)
            frame = frame * window
            frames.append(frame[None, :])
        frames = np.concatenate(frames, axis=0)
        return frames

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        if torch.is_tensor(wav):
            wav = wav.data.numpy().astype(np.float32)
        max_frames = wav.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            X = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            X = X[:, beg_i:end_i]
            pkg[self.name] = X
        else:
            wav = self.frame_signal(wav, self.window)
            #print('wav shape: ', wav.shape)
            lpc = pysptk.sptk.lpc(wav, order=self.order)
            #print('lpc: ', lpc.shape)
            pkg[self.name] = torch.FloatTensor(lpc)
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(order={}, hop={}, win={})'.format(self.order,
                                                    self.hop,
                                                    self.win)
        return self.__class__.__name__ + attrs

class MFCC(object):

    def __init__(self, n_fft=2048, hop=160,
                 order=13, sr=16000, win=400,
                 der_order=2, name='mfcc'):
        self.hop = hop
        # Santi: the librosa mfcc api does not always
        # accept a window argument, so we enforce n_fft
        # to be window to ensure the window len restriction
        #self.win = win
        self.n_fft = win
        self.order = order
        self.sr = 16000
        self.der_order=der_order
        self.name = name

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        y = wav.data.numpy()
        max_frames = y.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            mfcc = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            mfcc = mfcc[:, beg_i:end_i]
            pkg[self.name] = mfcc
        else:
            # print(y.dtype)
            mfcc = librosa.feature.mfcc(y, sr=self.sr,
                                        n_mfcc=self.order,
                                        n_fft=self.n_fft,
                                        hop_length=self.hop,
                                        #win_length=self.win,
                                        )[:, :max_frames]
            if self.der_order > 0 :
                deltas=[mfcc]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(mfcc,order=n))
                mfcc=np.concatenate(deltas)
    
            pkg[self.name] = torch.tensor(mfcc.astype(np.float32))
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(order={}, sr={})'.format(self.order,
                                           self.sr)
        return self.__class__.__name__ + attrs

class MFCC_librosa(object):

    def __init__(self, n_fft=2048, hop=160,
                 order=13, sr=16000, win=400,der_order=2,n_mels=40,
                 htk=True, name='mfcc_librosa'):
        self.hop = hop
        # Santi: the librosa mfcc api does not always
        # accept a window argument, so we enforce n_fft
        # to be window to ensure the window len restriction
        #self.win = win
        self.n_fft = win
        self.order = order
        self.sr = 16000
        self.der_order=der_order
        self.n_mels=n_mels
        self.htk=True
        self.name = name

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        y = wav.data.numpy()
        max_frames = y.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            mfcc = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            mfcc = mfcc[:, beg_i:end_i]
            pkg[self.name] = mfcc
        else:
            # print(y.dtype)
            mfcc = librosa.feature.mfcc(y, sr=self.sr,
                                        n_mfcc=self.order,
                                        n_fft=self.n_fft,
                                        hop_length=self.hop,
                                        #win_length=self.win,
					n_mels=self.n_mels,
                                        htk=self.htk,
                                        )[:, :max_frames]
            if self.der_order > 0 :
                deltas=[mfcc]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(mfcc,order=n))
                mfcc=np.concatenate(deltas)

            pkg[self.name] = torch.tensor(mfcc.astype(np.float32))
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(order={}, sr={})'.format(self.order,
                                           self.sr)
        return self.__class__.__name__ + attrs

class KaldiFeats(object):
    def __init__(self, kaldi_root, hop=160, win=400, sr=16000):

        if kaldi_root is None and 'KALDI_ROOT' in os.environ:
            kaldi_root = os.environ['KALDI_ROOT']

        assert kaldi_root is not None, (
            "Set KALDI_ROOT (either pass via cmd line, or set env variable)"
        )

        self.kaldi_root = kaldi_root
        self.hop = hop
        self.win = win
        self.sr = sr

        self.frame_shift = int(1000./self.sr * self.hop) #in ms
        self.frame_length = int(1000./self.sr * self.win) #in ms

    def __execute_command__(self, datain, cmd):
        #try:
        fin, fout = kio.open_or_fd(cmd, 'wb')
        kio.write_wav(fin, datain, self.sr, key='utt')
        fin.close() #so its clear nothing new arrives
        feats_ark = kio.read_mat_ark(fout)
        for _, feats in feats_ark:
            fout.close()
            return feats.T #there is only one to read
        #except Exception as e:
        #    print (e)
        #    return None

    def __repr__(self):
        return self.__class__.__name__

class KaldiMFCC(KaldiFeats):
    def __init__(self, kaldi_root, hop=160, win=400, sr=16000,
                    num_mel_bins=40, num_ceps=13, der_order=2,
                    name='kaldimfcc'):

        super(KaldiMFCC, self).__init__(kaldi_root=kaldi_root, 
                                        hop=hop, win=win, sr=sr)

        self.num_mel_bins = num_mel_bins
        self.num_ceps = num_ceps
        self.der_order=der_order

        cmd = "ark:| {}/src/featbin/compute-mfcc-feats --print-args=false "\
               "--use-energy=false --snip-edges=false --num-ceps={} "\
               "--frame-length={} --frame-shift={} "\
               "--num-mel-bins={} --sample-frequency={} ark:- ark:- |"\
               " {}/src/featbin/add-deltas --print-args=false "\
               "--delta-order={} ark:- ark:- |"

        self.cmd = cmd.format(self.kaldi_root, self.num_ceps,
                              self.frame_length, self.frame_shift,
                              self.num_mel_bins, self.sr, self.kaldi_root,
                              self.der_order)
        self.name = name

    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        y = wav.data.numpy()
        max_frames = y.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            mfcc = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            mfcc = mfcc[:, beg_i:end_i]
            pkg[self.name] = mfcc
        else:
            # print(y.dtype)
            mfccs = self.__execute_command__(y, self.cmd)
            assert mfccs is not None, (
                "Mfccs extraction failed"
            )
            pkg[self.name] = torch.tensor(mfccs[:,:max_frames].astype(np.float32))

        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = "(bins={}, ceps={}, sr={})"\
                  .format(self.num_mel_bins, self.num_ceps, self.sr)
        return self.__class__.__name__ + attrs

class KaldiPLP(KaldiFeats):
    def __init__(self, kaldi_root, hop=160, win=400, sr=16000,
                 num_mel_bins=20, num_ceps=20, lpc_order=20,
                 name='kaldiplp'):

        super(KaldiPLP, self).__init__(kaldi_root=kaldi_root, 
                                        hop=hop, win=win, sr=sr)

        self.num_mel_bins = num_mel_bins
        self.num_ceps = num_ceps
        self.lpc_order = lpc_order

        cmd = "ark:| {}/src/featbin/compute-plp-feats "\
               "--print-args=false --snip-edges=false --use-energy=false "\
               "--num-ceps={} --lpc-order={} "\
               "--frame-length={} --frame-shift={} "\
               "--num-mel-bins={} --sample-frequency={} "\
               "ark:- ark:- |"

        self.cmd = cmd.format(self.kaldi_root, self.num_ceps, self.lpc_order, 
                              self.frame_length, self.frame_shift, 
                              self.num_mel_bins, self.sr)
        self.name = name

    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        y = wav.data.numpy()
        max_frames = y.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            plp = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            plp = plp[:, beg_i:end_i]
            pkg[self.name] = plp
        else:
            # print(y.dtype)
            feats = self.__execute_command__(y, self.cmd)
            pkg[self.name] = torch.tensor(feats[:,:max_frames].astype(np.float32))
        
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = "(bins={}, ceps={}, sr={}, lpc={})"\
                  .format(self.num_mel_bins, self.num_ceps, self.sr, self.lpc_order)
        return self.__class__.__name__ + attrs


class Prosody(object):

    def __init__(self, hop=160, win=320, f0_min=60, f0_max=300,der_order=2,
                 sr=16000, name='prosody'):
        self.hop = hop
        self.win = win
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sr = sr
        self.der_order = der_order
        self.name = name

    # @profile
    def __call__(self, pkg, cached_file=None):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy()
        max_frames = wav.shape[0] // self.hop
        if cached_file is not None:
            # load pre-computed data
            proso = torch.load(cached_file)
            beg_i = pkg['chunk_beg_i'] // self.hop
            end_i = pkg['chunk_end_i'] // self.hop
            proso = proso[:, beg_i:end_i]
            pkg[self.name] = proso
        else:
            # first compute logF0 and voiced/unvoiced flag
            # f0 = pysptk.rapt(wav.astype(np.float32),
            #                 fs=self.sr, hopsize=self.hop,
            #                 min=self.f0_min, max=self.f0_max,
            #                 otype='f0')
            f0 = pysptk.swipe(wav.astype(np.float64),
                              fs=self.sr, hopsize=self.hop,
                              min=self.f0_min,
                              max=self.f0_max,
                              otype='f0')
            # sound = pm.Sound(wav.astype(np.float32), self.sr)
            # f0 = sound.to_pitch(self.hop / 16000).selected_array['frequency']
            if len(f0) < max_frames:
                pad = max_frames - len(f0)
                f0 = np.concatenate((f0, f0[-pad:]), axis=0)
            lf0 = np.log(f0 + 1e-10)
            lf0, uv = interpolation(lf0, -1)
            lf0 = torch.tensor(lf0.astype(np.float32)).unsqueeze(0)[:, :max_frames]
            uv = torch.tensor(uv.astype(np.float32)).unsqueeze(0)[:, :max_frames]
            if torch.sum(uv) == 0:
                # if frame is completely unvoiced, make lf0 min val
                lf0 = torch.ones(uv.size()) * np.log(self.f0_min)
            # assert lf0.min() > 0, lf0.data.numpy()
            # secondly obtain zcr
            zcr = librosa.feature.zero_crossing_rate(y=wav,
                                                     frame_length=self.win,
                                                     hop_length=self.hop)
            zcr = torch.tensor(zcr.astype(np.float32))
            zcr = zcr[:, :max_frames]
            # finally obtain energy
            egy = librosa.feature.rmse(y=wav, frame_length=self.win,
                                       hop_length=self.hop,
                                       pad_mode='constant')
            egy = torch.tensor(egy.astype(np.float32))
            egy = egy[:, :max_frames]
            proso = torch.cat((lf0, uv, egy, zcr), dim=0)
  
            if self.der_order > 0 :
                deltas=[proso]
                for n in range(1,self.der_order+1):
                    deltas.append(librosa.feature.delta(proso.numpy(),order=n))
                proso=torch.from_numpy(np.concatenate(deltas))

            pkg[self.name] = proso
        # Overwrite resolution to hop length
        pkg['dec_resolution'] = self.hop
        return pkg

    def __repr__(self):
        attrs = '(hop={}, win={}, f0_min={}, f0_max={}'.format(self.hop,
                                                               self.win,
                                                               self.f0_min,
                                                               self.f0_max)
        attrs += ', sr={})'.format(self.sr)
        return self.__class__.__name__ + attrs


class ZNorm(object):

    def __init__(self, stats):
        self.stats_name = stats
        with open(stats, 'rb') as stats_f:
            self.stats = pickle.load(stats_f)

    # @profile
    def __call__(self, pkg, ignore_keys=[]):
        pkg = format_package(pkg)
        for k, st in self.stats.items():
            # assert k in pkg, '{} != {}'.format(list(pkg.keys()),
            #                                   list(self.stats.keys()))
            if k in ignore_keys:
                continue
            if k in pkg:
                mean = st['mean'].unsqueeze(1)
                std = st['std'].unsqueeze(1)
                pkg[k] = (pkg[k] - mean) / std
        return pkg

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.stats_name)



class CachedCompose(Compose):

    def __init__(self, transforms, keys, cache_path):
        super().__init__(transforms)
        self.cache_path = cache_path
        self.keys = keys
        assert len(keys) == len(transforms), '{} != {}'.format(len(keys),
                                                               len(transforms))
        print('Keys: ', keys)

    def __call__(self, x):
        if 'uttname' not in x:
            raise ValueError('Utterance name not found when '
                             'looking for cached transforms')
        if 'split' not in x:
            raise ValueError('Split name not found when '
                             'looking for cached transforms')

        znorm_ignore_flags = []
        # traverse the keys to look for cache sub-folders
        for key, t in zip(self.keys, self.transforms):
            if key == 'totensor' or key == 'chunk':
                x = t(x)
            elif key == 'znorm':
                x = t(x, znorm_ignore_flags)
            else:
                aco_dir = os.path.join(self.cache_path, x['split'], key)
                if os.path.exists(aco_dir):
                    # look for cached file by name
                    bname = os.path.splitext(os.path.basename(x['uttname']))[0]
                    acofile = os.path.join(aco_dir, bname + '.' + key)
                    if not os.path.exists(acofile):
                        acofile = None
                    else:
                        znorm_ignore_flags.append(key)
                    x = t(x, cached_file=acofile)
        return x

    def __repr__(self):
        return super().__repr__()
