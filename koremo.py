import numpy as np
import librosa

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
set_session(tf.Session(config=config))
import keras.backend as K

from keras.models import load_model
mse_crs = load_model('model/total_s_rmse_cnn_rnnself_re-19-0.9723-0.9701.hdf5', compile=False)

def make_data(filenames, mlen = 500):
  fnum = len(filenames)
  data             = np.zeros((fnum,mlen,128))
  data_conv        = np.zeros((fnum,mlen,128,1))
  data_rmse        = np.zeros((fnum,mlen,1))
  data_s_rmse      = np.zeros((fnum,mlen,129))
  data_s_rmse_conv = np.zeros((fnum,mlen,129,1))
  for i in range(fnum):
    j = i + 1
    if j % 100 == 0:
      print(j)
    filename = '../data/' + filenames[i]
    y, sr = librosa.load(filename)
    D = np.abs(librosa.stft(y))**2
    ss, phase = librosa.magphase(librosa.stft(y))
    rmse = librosa.feature.rmse(S=ss)
    rmse = rmse/np.max(rmse)
    rmse = np.transpose(rmse)
    S = librosa.feature.melspectrogram(S=D)
    S = np.transpose(S)
    Srmse = np.multiply(rmse,S)
    if len(S)>=mlen:
      data[i][:,:]=S[-mlen:,:]
      data_conv[i][:,:,0]=S[-mlen:,:]
      data_rmse[i][:,0]=rmse[-mlen:,0]
      data_s_rmse[i][:,0]=rmse[-mlen:,0]
      data_s_rmse[i][:,1:]=S[-mlen:,:]
      data_s_rmse_conv[i][:,0,0]=rmse[-mlen:,0]
      data_s_rmse_conv[i][:,1:,0]=S[-mlen:,:]
    else:
      data[i][-len(S):,:]=S
      data_conv[i][-len(S):,:,0]=S
      data_rmse[i][-len(S):,0]=np.transpose(rmse)
      data_s_rmse[i][-len(S):,0]=np.transpose(rmse)
      data_s_rmse[i][-len(S):,1:]=S
      data_s_rmse_conv[i][-len(S):,0,0]=np.transpose(rmse)
      data_s_rmse_conv[i][-len(S):,1:,0]=S
  return data,data_conv,data_rmse,data_s_rmse,data_s_rmse_conv

def pred_emo(filenames):
  data,data_conv,data_rmse,data_s_rmse,data_s_rmse_conv =make_data(filenames)
  att_source= np.zeros((len(filenames),64))
  zs = np.asarray(mse_crs.predict([data_s_rmse_conv,data_s_rmse,att_source]))
  ys = np.argmax(zs,axis=1)
  # if y==0:
  #   print(">> Angry")
  # if y==1:
  #   print(">> Fear")
  # if y==2:
  #   print(">> Joy")
  # if y==3:
  #   print(">> Normal")
  # if y==4:
  #   print(">> Sad")
  return ys


