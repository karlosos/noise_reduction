from prediction_denoise import prediction


def predict(name_model='model_unet', audio_input_prediction='noisy_voice_long.wav',
            audio_output_prediction='denoise_t2.wav', sr=8000):
    # Example: python main.py --mode="prediction"
    # path to find pre-trained weights / save models
    weights_path = './data/weights'
    # pre trained model
    name_model = name_model
    # directory where read noisy sound to denoise
    audio_dir_prediction = './demo_data/test'
    # directory to save the denoise sound
    dir_save_prediction = './demo_data/save_predictions/'
    # Name noisy sound file to denoise
    audio_input_prediction = [audio_input_prediction]
    # Name of denoised sound file to save
    audio_output_prediction = audio_output_prediction
    # Sample rate to read audio
    sample_rate = sr
    # Minimum duration of audio files to consider
    min_duration = 1.0
    # Frame length for training data
    frame_length = 8064
    # hop length for sound files
    hop_length_frame = 8064
    # nb of points for fft(for spectrogram computation)
    n_fft = 255
    # hop length for fft
    hop_length_fft = 63

    prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
               audio_output_prediction, sample_rate, min_duration, frame_length, hop_length_frame, n_fft,
               hop_length_fft)


if __name__ == '__main__':
    predict()
