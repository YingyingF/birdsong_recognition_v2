## Data preprocessing
- To handle audio files of different length:
    - Split longer files into shorter segments. Since audio recordings are of very different length from 10s to several minutes. Can't simply split the train/test data by audio files. Either weight the classes, or oversample the class with shorter overall audio recordings.
    - Loop back the shorter segments to match the longer files
- Data augmentation:
    - Time or frequency masking
    - time warping
    - mixing bird songs from multiple species? but with different volumn? Many of the erros have multiple birds in the recording.
    - add another class for silence/noise. It seems the current classifier thinks silence/noise is blue jay..


## ML model
- Using Spectrogram as input, the architecture consists of a stack of CNN layers, followed by fully connected layers.

## Hyperparameter tuning
- Using weights & bias

## Error analysis
- Notice the issue that if using the approach of splitting longer files into shorter ones, some of the segments essentially only has noise. 
- If the model is not certain about the prediction, output more than one likely outputs? This also handles the case when there're multiple birds in the recording. 

## App