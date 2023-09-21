#이것은 mel-spectrogram + chroma 총 (audio,52,350) cnn으로 5폴드
#pip install librosa==0.9.2      1.스펙트로그램 + 크로마   2.mfcc filterbank 조절
#최종 배열에서 라벨마다 threshold값을 설정해서 라벨을 뽑는다.
# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import tensorflow as tf
from tqdm import tqdm
from glob import glob
import librosa
import librosa.display as dsp
import IPython.display as ipd
import time
def testfuc():

  warnings.filterwarnings(action='ignore')
  import random

  def seed_everything(seed):
      random.seed(seed)
      os.environ['PYTHONHASHSEED'] = str(seed)
      np.random.seed(seed)
      tf.random.set_seed(seed)

  seed_everything(813)  

  test = pd.read_csv('test.csv')

  sr = 20000
  size = 40
  pad_size = 350 #max=543 근데 이거는 수정이 필요할지도? max 찾아서 바꾸는걸로
  repeat_size = 5 #수정

  from tqdm.notebook import tqdm

  test_file_names = test["file_name"].to_numpy()

  def load_audio(file_names, target, path): 
    audios = []
    for audio in tqdm(file_names):
      # librosa를 이용하여 데이터 로드
      an_audio, _ = librosa.load(path+audio, sr=sr)
      audio_array = np.array(an_audio)
      audios.append(audio_array)
    audios = np.array(audios)

    return audios

  audio_test = load_audio(test_file_names, np.array([None]), path='./data/')


  def IAV(waveform,sample_rate):

      # 윈도우 크기와 타임 스텝 설정
      window_size = int(0.128 * sample_rate)  # 윈도우 크기 (샘플 수) 250
      hop_size = int(0.032 * sample_rate)  # hop 크기 (샘플 수)
 
      # 음성을 프레임으로 분할
      num_frames = (len(waveform) - window_size) // hop_size + 1
      frame_indices = [(i * hop_size, i * hop_size + window_size) for i in range(num_frames)]

      # IAV 특징 벡터 추출
      iav_features = []
      # 프레임 추출
      frames = [waveform[start:end] for start, end in frame_indices]
      # IAV 특징 벡터 추출
      iav_features = np.sum(np.abs(frames), axis=1) / window_size
    
      # 최댓값과 최솟값 계산
      max_value = np.max(iav_features)
      min_value = np.min(iav_features)

      # 최솟값 설정
      min_threshold = (max_value - min_value) * 0.1 + min_value
      if min_threshold > max_value * 0.7:
          min_threshold = max_value * 0.2

      # 음성 구간 추출
      speech_segments = []
      is_speech = False

      for i in range(len(iav_features)):
          if iav_features[i] > min_threshold:
              if not is_speech:
                  start_index = frame_indices[i][0]
                  is_speech = True
          else:
              if is_speech:
                  end_index = frame_indices[i-1][1]
                  is_speech = False
                  speech_segments.append((start_index, end_index))

      # 추출된 음성 구간을 하나의 배열로 연결
      concatenated_audio = np.concatenate([waveform[start:end] for start, end in speech_segments])
      # 새로운 WAV 파일로 저장
      return concatenated_audio

  prepro = []

  for i in range(len(audio_test)):
      preprocessed_segments = IAV(audio_test[i],sr)
      prepro.append(preprocessed_segments)

  audio_test = prepro 

  def random_pad(mels, pad_size, chroma=True):

    pad_width = pad_size - mels.shape[1]
    rand = np.random.rand()
    left = int(pad_width * rand)
    right = pad_width - left

    if chroma:
      mels = np.pad(mels, pad_width=((0,0), (left, right)), mode='constant')
      local_max, local_min = mels.max(), mels.min()
      mels = (mels - local_min)/(local_max - local_min)
    else: #mels
      local_max, local_min = mels.max(), mels.min()
      mels = (mels - local_min)/(local_max - local_min)
      mels = np.pad(mels, pad_width=((0,0), (left, right)), mode='constant')

    return mels

  #test audio에서 mell,mfcc추출, padding(이거는 1번)
  audio_mels_array_test = []
  audio_chroma_array_test = []

  for y in audio_test:
    mels = librosa.feature.melspectrogram(y, sr=sr, n_mels=size)
    mels = librosa.power_to_db(mels, ref=np.max)

    chroma = librosa.feature.chroma_stft(y, sr=sr, hop_length=512)

    audio_mels_array_test.append(random_pad(mels, pad_size=pad_size, chroma=False))
    audio_chroma_array_test.append(random_pad(chroma, pad_size=pad_size, chroma=True))

  audio_mels_array_test = np.array(audio_mels_array_test, np.float64)
  audio_chroma_array_test = np.array(audio_chroma_array_test, np.float64)
  audio_features_test = np.concatenate((audio_mels_array_test, audio_chroma_array_test), axis=1)

  #test
  from tensorflow.keras.models import load_model
  from sklearn.metrics import accuracy_score

  pred_list = []

  for fold in range(5):

    print(f'\n********** {fold+1} fold **********')

    preds_val_list = []
    ### melspectrogram ###
    filepath = f"model.mels_{fold}.hdf5"
    model = load_model(filepath)
    pred_list.append(model.predict(audio_features_test)) #test예측

  ### ensemble ###
  test_pred_result = np.zeros((1, 4))
  for i in range(0, len(pred_list)):
    test_pred_result += pred_list[i]

  total_sum = np.sum(test_pred_result)
  # 정규화
  normalized_result = test_pred_result / total_sum
  normalized_result = np.round(normalized_result, 3)

  return normalized_result

