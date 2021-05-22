## Data preprocessing
- To handle audio files of different length:
    - Split longer files into shorter segments
    - Loop back the shorter segments to match the longer files
- Data augmentation:
    - Time or frequency masking
    - time warping

## ML model
- Using Spectrogram as input, the architecture consists of a stack of CNN layers, followed by fully connected layers.

## Hyperparameter tuning
- Using weights & bias

## Error analysis
- Notice the issue that if using the approach of splitting longer files into shorter ones, some of the segments essentially only has noise. 

## App