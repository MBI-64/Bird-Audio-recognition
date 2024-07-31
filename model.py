from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

import os
from matplotlib import pyplot as plt
import tensorflow as tf
import librosa
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


BIRD_FILE = os.path.join('newdata', 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
ENV_FILE = os.path.join('newdata', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')

def load_wav(filename):
    filename = filename.numpy().decode('utf-8')  # Decode bytes to string
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav

def tf_load_wav(filename):
    return tf.py_function(load_wav, [filename], tf.float32)

def load_wav_wrapper(filename):
    wav = tf.py_function(load_wav, [filename], tf.float32)
    wav.set_shape([None])
    return wav

bird_wave = load_wav_wrapper(tf.constant(BIRD_FILE))
env_wave = load_wav_wrapper(tf.constant(ENV_FILE))

plt.plot(bird_wave.numpy())
plt.plot(env_wave.numpy())
plt.show()

# Create Tensorflow Dataset
POS = os.path.join('newdata', 'Parsed_Capuchinbird_Clips')
NEG = os.path.join('newdata', 'Parsed_Not_Capuchinbird_Clips')

pos = tf.data.Dataset.list_files(POS + '/*.wav')
neg = tf.data.Dataset.list_files(NEG + '/*.wav')

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(list(pos))))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(list(neg))))))

data = positives.concatenate(negatives)

# Average length of a Capuchin call
lengths = []
for file in os.listdir(os.path.join('newdata', 'Parsed_Capuchinbird_Clips')):
    tensor_wave = load_wav_wrapper(tf.constant(os.path.join('newdata', 'Parsed_Capuchinbird_Clips', file)))
    lengths.append(len(tensor_wave))

print("Mean Length:", tf.math.reduce_mean(lengths).numpy())
print("Min Length:", tf.math.reduce_min(lengths).numpy())
print("Max Length:", tf.math.reduce_max(lengths).numpy())

# Convert to Spectrogram
def preprocess(file_path, label):
    wav = tf_load_wav(file_path)
    wav.set_shape([None])
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)

plt.figure(figsize=(15, 10))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()

data = data.map(preprocess)
data = data.shuffle(buffer_size=500)
data = data.batch(4)
data = data.prefetch(2)  

train = data.take(30)
test = data.skip(30).take(15)

samples, labels = train.as_numpy_iterator().next()
print(samples.shape)

# Build deep learning model
model = Sequential()
model.add(Conv2D(8, (3,3), activation='relu', input_shape=(1491, 257, 1)))
model.add(Conv2D(8, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu')) 
model.add(Dense(1, activation='sigmoid'))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.summary()

# Save the best model 
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, mode='min')
callbacks_list = [checkpoint]

hist = model.fit(train, epochs=50, validation_data=test, callbacks=callbacks_list)

plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.show()

X_test, y_test = test.as_numpy_iterator().next()
yhat = model.predict(X_test)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

def load_mp3(filename):
    filename = filename.numpy().decode('utf-8') 
    wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
    return wav

def load_mp3_wrapper(filename):
    wav = tf.py_function(load_mp3, [filename], tf.float32)
    wav.set_shape([None])
    return wav

mp3 = os.path.join('newdata', 'Forest Recordings', 'recording_00.mp3')
wav = load_mp3_wrapper(tf.constant(mp3))

# Define preprocess_mp3 function
def preprocess_mp3(sample, index):
    sample = sample[0]  
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(16) 

yhat = model.predict(audio_slices)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

# Group consecutive detections
from itertools import groupby

yhat = [key for key, group in groupby(yhat)]
calls = tf.math.reduce_sum(yhat).numpy()

print("Number of calls in the audio is: ", calls)

# Make predictions
results = {}
for file in os.listdir(os.path.join('newdata', 'Forest Recordings')):
    FILEPATH = os.path.join('newdata', 'Forest Recordings', file)

    wav = load_mp3_wrapper(tf.constant(FILEPATH))
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    
    yhat = model.predict(audio_slices)
    
    results[file] = yhat

print(results)

# Convert predictions into classes
class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]
class_preds

# Group consecutive predictions
postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
postprocessed

with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Writing to model.json file")
