import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import model_from_json
from signal_utils import scaled_in, inv_scaled_ou
from signal_utils import (
    audio_files_to_numpy,
    numpy_audio_to_matrix_spectrogram,
    matrix_spectrogram_to_numpy_audio,
    save_audio,
)

weights_path = "./data/weights"
# pre trained model
name_model = "model_unet"
sample_rate = 8000
min_duration = 1.0
frame_length = 8064
hop_length_frame = 8064

json_file = open(weights_path + "/" + name_model + ".json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(weights_path + "/" + name_model + ".h5")
print("Loaded model from disk")

# Extracting noise and voice from folder and convert to numpy
# audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, sample_rate,
#                              frame_length, hop_length_frame, min_duration)

# Dimensions of squared spectrogram
n_fft = 255
hop_length_fft = 63
dim_square_spec = int(n_fft / 2) + 1
print(dim_square_spec)


def print_sound(indata, outdata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    print(volume_norm)
    m_mag_db = np.zeros((1, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((1, dim_square_spec, dim_square_spec), dtype=complex)

    stftaudio = librosa.stft(indata[:, 0], n_fft=n_fft, hop_length=hop_length_fft)
    stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)
    m_phase[0, :, :] = stftaudio_phase
    m_mag_db[0, :, :] = librosa.amplitude_to_db(stftaudio_magnitude, ref=np.max)

    X_in = scaled_in(m_mag_db)
    # # Reshape for prediction
    X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
    # # Prediction using loaded network
    X_pred = loaded_model.predict(X_in)
    # # # Rescale back the noise model
    inv_sca_X_pred = inv_scaled_ou(X_pred)
    # # # Remove noise model from noisy speech
    X_denoise = m_mag_db - inv_sca_X_pred[:, :, :, 0]

    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(
        X_denoise, m_phase, frame_length, hop_length_fft
    )

    outdata[:, 0] = audio_denoise_recons * 2
    outdata[:, 1] = audio_denoise_recons * 2


with sd.Stream(samplerate=8000, blocksize=8064, callback=print_sound):
    sd.sleep(100000)
