import matplotlib.pyplot as plt
import librosa.display


def plot_spectrogram(stftaudio_magnitude_db,sample_rate, hop_length_fft) :
    """This function plots a spectrogram"""
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(stftaudio_magnitude_db, x_axis='time', y_axis='linear',
                             sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    title = 'hop_length={},  time_steps={},  fft_bins={}  (2D resulting shape: {})'
    plt.title(title.format(hop_length_fft,
                           stftaudio_magnitude_db.shape[1],
                           stftaudio_magnitude_db.shape[0],
                           stftaudio_magnitude_db.shape))
    return


def plot_3_spectograms(stftvoicenoise_mag_db,stftnoise_mag_db,stftvoice_mag_db,sample_rate, hop_length_fft):
    """This function plots the spectrograms of noisy voice, noise and voice as a single plot """
    plt.figure(figsize=(8, 12))
    plt.subplot(3, 1, 1)
    plt.title('Spectrogram voice + noise')
    librosa.display.specshow(stftvoicenoise_mag_db, x_axis='time', y_axis='linear',sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.title('Spectrogram predicted voice')
    librosa.display.specshow(stftnoise_mag_db, x_axis='time', y_axis='linear',sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.title('Spectrogram true voice')
    librosa.display.specshow(stftvoice_mag_db, x_axis='time', y_axis='linear',sr=sample_rate, hop_length=hop_length_fft)
    plt.colorbar()
    plt.tight_layout()

    return