from birdsong_recognition.utils import add_channel_dim, get_sample_labels, get_spectrogram, load_mp3, preprocess_file, wrapper_split_file_by_window_size
import tensorflow as tf
import os
import numpy as np

EBIRDS = ['norcar', 'blujay', 'bkcchi']

### reload trained model
model = tf.keras.models.load_model('./model/model.h5')

### get an audio sample
audio_dir = './dataset/'
sample_label = np.random.choice(EBIRDS)
sample_label_all_files = os.listdir(os.path.join(audio_dir, sample_label))
sample_audio = np.random.choice(sample_label_all_files)
sample_audio = os.path.join(audio_dir, sample_label, sample_audio)

print(sample_audio)

### preprocess the audio 
sample_audio_label= get_sample_labels([sample_audio], EBIRDS)
decoded_audio, label = load_mp3(sample_audio_label[0])
decoded_audio, label = preprocess_file(decoded_audio, label)
decoded_audio, label = wrapper_split_file_by_window_size(decoded_audio, label)
decoded_audio, label = get_spectrogram(decoded_audio, label)
decoded_audio, label = add_channel_dim(decoded_audio, label)

### Predict
predict = model.predict(decoded_audio)
predict_category = predict.argmax(axis=1)
pred_idx, pred_count = np.unique(predict_category, return_counts=True)
majority_prediction = pred_idx[pred_count.argmax()]
predicted_bird = EBIRDS[majority_prediction]
confidence_level = pred_count.max()/len(predict_category)

print('True label: {}, predicted bird: {}.'.format(sample_label, predicted_bird))
print('confidence level: {}'.format(confidence_level))

print('done')