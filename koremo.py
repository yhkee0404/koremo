import numpy as np
import librosa

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
set_session(tf.Session(config=config))
import keras.backend as K

def make_x(filenames, mlen=500):
  fnum = len(filenames)
  shape = (fnum,mlen,129)
  data_s_rmse = np.zeros(shape)
  for i in range(fnum):
    j = i + 1
    filename = '../data/' + filenames[i]

    y, sr = librosa.load(filename)
    D = np.abs(librosa.stft(y))**2
    ss, phase = librosa.magphase(librosa.stft(y))
    rmse = librosa.feature.rmse(S=ss)
    rmse = (rmse/np.max(rmse)).T
    S = librosa.feature.melspectrogram(S=D).T

    data_s_rmse[i][-min(mlen, len(S)):]=np.concatenate((rmse, S), axis=1)[-mlen:]
  data_s_rmse_conv = data_s_rmse.view()
  data_s_rmse_conv.shape = shape + (1,)
  return [data_s_rmse_conv,data_s_rmse,np.zeros((fnum,64))]

def pred_emo(model, filenames):
  z = model.predict(make_x(filenames))
  y = np.argmax(z, axis=1)
  return y
  
def pred_data_s_rmse(model, filename, start=0, batch_size=500):
  import mmap, os

  filenames = os.listdir('../data')
  fnum = len(filenames)
  y_total = np.zeros(fnum)
  x = [0, 0, np.zeros((fnum,64))]
  for i in range(start, fnum, batch_size):
    min_batch_size = min(fnum - i, batch_size)
    j = i + min_batch_size
    with open('../data_s_rmse/' + str(j) + '.npy', 'rb') as f:
      mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

      version = np.lib.format.read_magic(f)
      np.lib.format._check_version(version)
      shape, fortran_order, dt = np.lib.format._read_array_header(f, version)

      if fortran_order:
        order = 'F'
      else:
        order = 'C'
      if dt.hasobject:
        msg = "Array can't be memory-mapped: Python objects in dtype."
        raise ValueError(msg)

      x[1] = np.ndarray(shape, dtype=np.dtype(dt), buffer=mm, offset=0, order=order)
      x[0] = x[1].view()
      x[0].shape = shape + (1,)
      
      z = np.asarray(model.predict(x))
      y = np.argmax(z,axis=1)
      
      y_total[i:j] = y

      mm.close()
  np.save(filename, y_total)

def save_data_s_rmse(start=0, batch_size=500, mlen=500):
  import os

  filenames = os.listdir('../data')
  filenames.sort()
  fnum = len(filenames)
  for i in range(start, fnum, batch_size):
    min_batch_size = min(fnum - i, batch_size)

    data_s_rmse = np.zeros((min_batch_size,mlen,129))
    for j in range(min_batch_size):
      k = i + j
      filename = '../data/' + filenames[k]

      y, sr = librosa.load(filename)
      D = np.abs(librosa.stft(y))**2
      ss, phase = librosa.magphase(librosa.stft(y))
      rmse = librosa.feature.rms(S=ss)
      rmse = (rmse/np.max(rmse)).T
      S = librosa.feature.melspectrogram(S=D).T

      data_s_rmse[j][-min(mlen, len(S)):]=np.concatenate((rmse, S), axis=1)[-mlen:]
    np.save('../data_s_rmse/' + str(i + min_batch_size), data_s_rmse)


