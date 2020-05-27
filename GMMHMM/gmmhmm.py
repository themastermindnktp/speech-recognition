"""
# GMMHMM for isolated word recognition
* HMM model for audio files of 6 categories ("khong", "nguoi", "viet_nam", "cua", "trong", "y_te")
* predicts corresponding labels on testing set
* visualize by confusion matrix
"""

# %%
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix
import operator
import itertools
#import scipy.stats as sp
from scipy.io import wavfile
import math
from python_speech_features import mfcc
import pickle
import soundfile as sf


# MFCC python_speech_features
def get_mfcc(audio_path):
    sr, y =  wavfile.read(audio_path)
    return mfcc(y, nfft=2048,winlen=0.025, winstep=0.01,\
                appendEnergy=True,winfunc=np.hamming, samplerate=sr, numcep=13)


def get_mfcc_(file_path):
    y, sr = librosa.load(file_path) # read .wav file
    hop_length = math.floor(sr*0.010) # 10ms hop
    win_length = math.floor(sr*0.025) # 25ms frame
    # mfcc is 12 x T matrix
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=2048,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1,1)) 
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 36 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 36 (transpose of X)
    return X.T # hmmlearn use T x N matrix


num_of_states = 12  
num_of_mix = 2  
covariance_type = 'diag'  # covariance type
n_iter = 1000  
dimension = 1


# Initial prior prob vector
def startProb():
    startProb = np.zeros(num_of_states)
    startProb[0: dimension] = 1/float(dimension)
    return startProb
# startProb()


# The initial transition matrix
def transitionMatrix():
    transmat_prior = (1 / float(dimension + 1)) * np.eye(num_of_states)

    for i in range(num_of_states - dimension):
        for j in range(dimension):
            transmat_prior[i, i + j + 1] = 1. / (dimension + 1)
    j = 0
    for i in range(num_of_states - dimension, num_of_states):
        for j in range(num_of_states - i - j):
            transmat_prior[i, i + j] = 1. / (num_of_states - i)

    return transmat_prior
# transitionMatrix()


#Construct GMM + HMM based on passed parameters (n_mix, transmat_prior, startprob_prior)
def GMMHMM() :
    return hmm.GMMHMM(n_components = num_of_states, n_mix = num_of_mix,\
                      verbose=True,\
                      transmat_prior = transitionMatrix(), startprob_prior = startProb(), \
                      covariance_type = covariance_type, n_iter = n_iter)


#Construct Gaussian HMM, i.e. GMM + HMM  1 mixture model
def GaussianHMM() :
    return hmm.GaussianHMM(n_components = num_of_states, \
                           verbose=True, \
                           #transmat_prior = transitionMatrix(), startprob_prior = startProb(), \
                           covariance_type = covariance_type, n_iter = n_iter)


labels = []
words = [] 
features = []
hmmModels = []
folder_dir = 'SpeechData' # Folder directory of the dataset, config this 


# for file_name in os.listdir(folder_dir):
#     features = np.array([])
#     data_length = len(os.listdir(folder_dir + '/' + file_name))
#     # Index to split data into two parts train/test (.8/.2) 
#     training_index = int(data_length * 0.8)
#     # loop and compute MFCC
#     for audio_name in os.listdir(folder_dir + '/' + file_name)[0:training_index]:#train from 0 to index
#         if len(features) == 0:
#             features = get_mfcc(folder_dir + '/' + file_name + '/' + audio_name)
#         else:
#             features = np.append(features, get_mfcc(folder_dir + '/' + file_name + '/' + audio_name), axis=0)
#         labels.append(file_name) # list labels (file name)
#         if file_name not in words:
#             words.append(file_name)
#     #  GMMHMM model with parameter
#     hmmModel = GMMHMM()
#     np.seterr(all='ignore')
#     print("---")
#     # Train hmm model on MFCC features 
#     hmmModel.fit(features)
#     name = ("GMMHMM" +file_name)
#     pickle.dump(hmmModel, open(name, 'wb'))
#     print('Finished training for: ', file_name)
#     hmmModels.append((hmmModel, file_name))# list models


# with open("Models_parameters.txt", "w") as f:
#     for i in hmmModels:
#         model, label = i
#         f.write(f"Model_name : {label}\n")
#         f.write("Initial state occupation distribution\n")
#         f.write(" ".join(map(str, model.startprob_)))
#         f.write("\nMatrix of transition probabilities between states\n")
#         f.write(" ".join(map(str, model.transmat_)))
#         f.write("\nMean parameters for each mixture component in each state\n")
#         f.write(" ".join(map(str, model.means_)))
#         f.write("\nCovariance parameters for each mixture components in each state\n")
#         f.write(" ".join(map(str, model.means_)))
#         f.write("\n\n")


# successful = 0
# predicted_labels = []
# real_labels = []
# for file_name in os.listdir(folder_dir):
#     data_length = len(os.listdir(folder_dir + '/' + file_name))
#     testing_index = int(data_length * 0.8)# index 
#     for audio_name in os.listdir(folder_dir + '/' + file_name)[testing_index:data_length]:# test from index to end
#         features = get_mfcc(folder_dir + '/' + file_name + '/' + audio_name)
#         probs = {}
#         for item in hmmModels:
#             hmm_model, label = item
#             # Calculate score of each observation sequence (log likelihood)
#             probs[label] = hmm_model.score(features)
#         # Get key having the highest score 
#         result = max(probs.items(), key=operator.itemgetter(1))[0]
#         predicted_labels.append(result)
#         real_labels.append(file_name)
#         if (result == file_name):
#             successful = successful + 1
# print('Accuracy on testing set : ', (successful * 100 / len(real_labels)))

# print("num of successful predict: ", successful)
# print("num of samples: ", len(real_labels))


# # Confusion matrix of testing set
# conf_matrix = confusion_matrix(real_labels, predicted_labels)

# #np.set_printoptions(precision=2)
# #plt.figure()
# conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
# conf_matrix
# plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion matrix')
# plt.xticks(range(len(words)), words, rotation=45)
# plt.yticks(range(len(words)), words)
# for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
#     plt.text(j, i, format(conf_matrix[i, j], '.2f'),
#              horizontalalignment="center",
#              color="white" if i == j else "black")
# #plt.tight_layout()
# plt.ylabel('Correct label')
# plt.xlabel('Predicted label')
# plt.show()


import pyaudio
import wave
from pydub import AudioSegment
from pydub.playback import play


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    trim_ms = 0
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms


def trim_audio(file_name):
    sound = AudioSegment.from_file(file_name, format="wav") 

    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)    
    trimmed_sound = sound[start_trim:duration-end_trim]
    
    trimmed_sound.export(file_name, format="wav")
    play(trimmed_sound)


def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "data/output.wav" # hard coding for now

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print ("* Listening...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print ("* Done Listening")

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Write the data to a wav file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


DATA_DIR = "/Users/tien/Downloads/hmm-word-recognition/data"


import librosa


# def run_recognition_system_demo(directory):
#     file_id = 'output'
#     record_audio() 
#     trim_audio()
#     mfcc = get_mfcc(os.path.join(DATA_DIR, file_id + '.wav'))
#     #y, sr = librosa.load(os.path.join(DATA_DIR, file_id + '.wav'))
#     #get_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39).transpose().reshape(-1,1)
#     mfcc = mfcc[0].reshape(-1, 1)
#     probs = {} # save score(log_prob)
#     labels = ["cua","khong","trong", "vietnam","yte","nguoi"]
#     for i in labels:
#         print(i)
#         filename = ("GMMHMM"+i)
#         #print(filename)
#         gmmhmm = pickle.load(open(filename, 'rb'))
#         probs[i]= gmmhmm.score(mfcc)
#         print(probs[i])
#     pred = max(probs.items(),key=operator.itemgetter(1))[0]
#     print("prediction: ", pred)

# # run_recognition_system_demo(DATA_DIR)
