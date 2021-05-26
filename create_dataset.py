import os
import numpy as np
from signal_utils import save_audio
from signal_utils import audio_files_to_numpy
from signal_utils import blend_voice_with_noise
from signal_utils import numpy_audio_to_matrix_spectrogram
import argparse


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def prepare_folders():
    make_dir('./data/train/timeseries/')
    make_dir('./data/train/combined_sound/')
    make_dir('./data/train/spectogram/')
    make_dir('./data/test/timeseries/')
    make_dir('./data/test/combined_sound/')
    make_dir('./data/test/spectogram/')


def create_data_from_folder(data_type):
    clean_voice_path = f"./data/{data_type}/clean_voice/"
    noise_path = f"./data/{data_type}/noise/"
    output_timeseries_path = f"./data/{data_type}/timeseries/"
    output_sound_path = f"./data/{data_type}/combined_sound/"
    output_spectogram = f"./data/{data_type}/spectogram/"
    create_data(clean_voice_path, noise_path, output_timeseries_path, output_sound_path, output_spectogram)


def create_data(clean_voice_path, noise_path, output_timeseries_path, output_sound_path, output_spectogram_path,
                sample_rate=8000,
                min_duration=1.0, frame_length=8064, jump_length_frame=8064, jump_length_frame_noise=5000,
                nb_samples=1000,
                n_fft=255, jump_length_fft=63):
    list_noise_files = os.listdir(noise_path)
    list_voice_files = os.listdir(clean_voice_path)

    # Convert signals to numpy arrays
    noise = audio_files_to_numpy(noise_path, list_noise_files, sample_rate,
                                 frame_length, jump_length_frame_noise, min_duration)

    voice = audio_files_to_numpy(clean_voice_path, list_voice_files,
                                 sample_rate, frame_length, jump_length_frame, min_duration)

    # Blending clean voice with noises (data augmentation)
    prod_voice, prod_noise, prod_noisy_voice = blend_voice_with_noise(
        voice, noise, nb_samples, frame_length)

    # Save generated sounds
    noisy_voice_long = prod_noisy_voice.reshape(1, nb_samples * frame_length)
    save_audio(output_sound_path + 'noisy_voice_long.wav', noisy_voice_long[0, :], sample_rate)
    voice_long = prod_voice.reshape(1, nb_samples * frame_length)
    save_audio(output_sound_path + 'voice_long.wav', voice_long[0, :], sample_rate)
    noise_long = prod_noise.reshape(1, nb_samples * frame_length)
    save_audio(output_sound_path + 'noise_long.wav', noise_long[0, :], sample_rate)

    # Spectrogram dimensions
    dim_square_spec = int(n_fft / 2) + 1

    # Create spectrograms
    m_amp_db_voice, m_pha_voice = numpy_audio_to_matrix_spectrogram(
        prod_voice, dim_square_spec, n_fft, jump_length_fft)
    m_amp_db_noise, m_pha_noise = numpy_audio_to_matrix_spectrogram(
        prod_noise, dim_square_spec, n_fft, jump_length_fft)
    m_amp_db_noisy_voice, m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
        prod_noisy_voice, dim_square_spec, n_fft, jump_length_fft)

    # Save to disk for training/testing
    np.save(output_timeseries_path + 'voice_timeserie', prod_voice)
    np.save(output_timeseries_path + 'noise_timeserie', prod_noise)
    np.save(output_timeseries_path + 'noisy_voice_timeserie', prod_noisy_voice)

    np.save(output_spectogram_path + 'voice_amp_db', m_amp_db_voice)
    np.save(output_spectogram_path + 'noise_amp_db', m_amp_db_noise)
    np.save(output_spectogram_path + 'noisy_voice_amp_db', m_amp_db_noisy_voice)

    np.save(output_spectogram_path + 'voice_pha_db', m_pha_voice)
    np.save(output_spectogram_path + 'noise_pha_db', m_pha_noise)
    np.save(output_spectogram_path + 'noisy_voice_pha_db', m_pha_noisy_voice)


if __name__ == '__main__':
    prepare_folders()

    parser = argparse.ArgumentParser(description='Prepare test or train dataset')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    test_flag = args.test
    train_flag = args.train

    if test_flag:
        print("Creating test dataset")
        create_data_from_folder("test")
    if train_flag:
        print("Creating train dataset")
        create_data_from_folder("train")
