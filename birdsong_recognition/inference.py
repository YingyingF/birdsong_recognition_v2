from birdsong_recognition.utils import add_channel_dim, get_sample_labels, get_spectrogram, load_mp3, preprocess_file, wrapper_split_file_by_window_size
import tensorflow as tf
import os
import numpy as np
from colorama import Style, Fore
import argparse
import numpy as np


def inference(input_audio):

    ### reload trained model
    model = tf.keras.models.load_model('./model/model.h5')

    ### get an audio sample
    #TODO: need to make get_sample_labels not dependent on folder than structure somehow.
    audio_dir = './test_data/norcar/'

    # sample_audio = np.random.choice(sample_label_all_files)
    sample_audio = os.path.join(audio_dir, input_audio)
    EBIRDS = ['norcar', 'blujay', 'bkcchi']
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

    print('Predicted bird: {}.'.format(predicted_bird))
    print('confidence level: %.3f'%confidence_level)
    return predicted_bird, np.round(confidence_level,3)


if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--file', type=str, required=True) 
   args = parser.parse_args()

   inference(args.file)
