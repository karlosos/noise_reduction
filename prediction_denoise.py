import librosa
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from signal_utils import scaled_in, inv_scaled_ou
from signal_utils import audio_files_to_numpy, numpy_audio_to_matrix_spectrogram, matrix_spectrogram_to_numpy_audio, save_audio


def prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
               audio_output_prediction, sample_rate, min_duration, frame_length, jump_length_frame, n_fft,
               jump_length_fft):
    """ This function takes as input pretrained weights, noisy voice sound to denoise, predict
    the denoise sound and save it to disk.
    """

    # load json and create model
    model_json = open(f'{weights_path}/{name_model}.json', 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(f'{weights_path}/{name_model}.h5')
    print("Loaded model from disk")

    # Extracting noise and voice from folder and convert to numpy
    audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate,
                                 frame_length, jump_length_frame, min_duration)

    # Dimensions of squared spectrogram
    dim_square_spec = int(n_fft / 2) + 1
    print(dim_square_spec)

    # Create Amplitude and phase of the sounds
    audio_amplitude, audio_phase = numpy_audio_to_matrix_spectrogram(
        audio, dim_square_spec, n_fft, jump_length_fft)

    # global scaling to have distribution -1/1
    X_scaled = scaled_in(audio_amplitude)

    # Reshape for prediction
    X_scaled = X_scaled.reshape(X_scaled.shape[0],X_scaled.shape[1],X_scaled.shape[2],1)

    # Prediction using loaded network
    X_pred = loaded_model.predict(X_scaled)

    # Rescale back the noise model
    X_pred_rescaled = inv_scaled_ou(X_pred)

    # Remove noise model from noisy speech
    X_denoised = audio_amplitude - X_pred_rescaled[:,:,:,0]

    # Reconstruct audio from denoised spectrogram and phase
    print(X_denoised.shape)
    print(audio_phase.shape)
    print(frame_length)
    print(jump_length_fft)

    audio_denoised = matrix_spectrogram_to_numpy_audio(X_denoised, audio_phase, frame_length, jump_length_fft)
    # Number of frames
    samples_ct = audio_denoised.shape[0]
    # Save all frames in one file
    denoise_long = audio_denoised.reshape(1, samples_ct * frame_length)*10
    save_audio(dir_save_prediction + audio_output_prediction, denoise_long[0, :], sample_rate)


def predict(audio_input_prediction, audio_output_prediction, sr=8000, name_model='model_unet',
            path='..\\data\\validation\\'):
    # path to find pre-trained weights / save models
    weights_path = '..\\data\\weights'
    # pre trained model
    name_model = name_model
    # directory where read noisy sound to denoise
    audio_dir_prediction = f'{path}noisy_voice'
    # directory to save the denoise sound
    dir_save_prediction = f'{path}save_prediction\\'
    # Name noisy sound file to denoise
    audio_input_prediction = [audio_input_prediction]
    # Name of denoised sound file to save
    audio_output_prediction = audio_output_prediction
    # Minimum duration of audio files to consider
    min_duration = 1.0
    # Frame length for training data
    frame_length = 8064
    # jump length for sound files
    jump_length_frame = 8064
    # nb of points for fft(for spectrogram computation)
    n_fft = 255
    # jump length for fft
    jump_length_fft = 63

    prediction(weights_path, name_model, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
               audio_output_prediction, sr, min_duration, frame_length, jump_length_frame, n_fft,
               jump_length_fft)