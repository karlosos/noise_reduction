<p align="center">
    <img src="https://i.imgur.com/PC1hnc0.png" width="300px" alt="logo"/>
</p>

***

<p align="center">
  <a href="#about">About</a> •
  <a href="#development">Development</a> •
  <a href="#references">References</a> •
  <a href="#contributors">Contributors</a>
</p>

<div align="center">

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub contributors](https://img.shields.io/github/contributors/karlosos/noise_reduction.svg)](https://github.com/karlosos/noise_reduction/graphs/contributors/)

</div>

## About

In this project, we are researching the possibility of noise reduction from voice signals using a deep neural network. We define the noise as sounds from computer fans, clicking on the keyboard, dog barking, or birds chirping.

Noise reduction is executed as the difference between input spectrogram and predicted noise spectrogram. We will predict noise spectrogram using CNN. U-Net was chosen as a backbone as we found that it was successfully used in [Singing Voice Separation with Deep U-Net Convolutional Networks](https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf).

Voice signals are available at [LibriSpeech](http://www.openslr.org/12/). LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project. Noise signals were collected from [ESC-50 dataset](https://github.com/karolpiczak/ESC-50). The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.

## Development

Clone repository:
```
git clone https://github.com/karlosos/noise_reduction 
cd noise_reduction
```

Create virtual environment, e.g. with `virtualenv`:

```
python -m virtualenv .venv
./venv/Script/activate
```

Install requirements:

```
pip install -r requirements.txt
```

Saving notebooks to html was done with following command:

```
jupyter nbconvert --to html notebooks/denoising_presentation.ipynb
```

Example html output can be accessed trough [htmlpreview.github.io](https://htmlpreview.github.io/?https://github.com/karlosos/noise_reduction/blob/main/notebooks/denoising_presentation.html).

### Data preparation

Data was downloaded and prepared using Google Colab notebook that is available in `./notebooks/prepare_data.ipynb` or [here](https://colab.research.google.com/drive/1pHQtifx5qlcXN34fEFxfluW4m2D8nM2x?usp=sharing) or on [nbviewer](https://nbviewer.jupyter.org/github/karlosos/noise_reduction/blob/main/notebooks/prepare_data.ipynb). Training data is not available for download as it took too much space on Google Drive. We created our dataset consisting of `dev-clean.tar.gz` (development set from LibriSpeech) and noises from ESC-50 from 3 categories (keyboard typing, mouse clicking, birds chirping).
Place `data/` folder into root of this repository.

### Training

Training was done using Google Colab and notebook that is available in `./notebooks/training.ipynb` or [here](https://colab.research.google.com/drive/1pJW6Tkqz56ZUqv7Nnzo7UBULBf8U7b_n#scrollTo=2letUDbSxQyn) or on [nbviewer](https://nbviewer.jupyter.org/github/karlosos/noise_reduction/blob/main/notebooks/training.ipynb). Training should be done in few iterations as Google is limiting available disk space. Training was done using Tensorflow and unet model. 

Goal of the model is to recreate noise spectrogram. Exemplary test and predicted spectrograms are shown below:

<img src="https://i.imgur.com/68mg7NW.png" width="350px" />

### Denoising

Denoising is done by subtracting input noisy voice spectrogram and predicted noise spectrogram. From clean spectrogram we can recreate timeseries and sound with inverse Short Time Fourier transform (iSTFT). This process is implemented in `prediction_denoise.py`.

### Denoised data presentation

Data visualisation and comparision is available in `./notebooks/denoising_presentation.ipynb` or [here](https://htmlpreview.github.io/?https://github.com/karlosos/noise_reduction/blob/main/notebooks/denoising_presentation.html) or on [nbviewer](https://nbviewer.jupyter.org/github/karlosos/noise_reduction/blob/main/notebooks/denoising_presentation.ipynb).

We have tested real data with keyboard typing, mouse clicking and birds chirping noises.

## References

1. [[jansson2017singing]](https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf) Jansson, Andreas, Eric J. Humphrey, Nicola Montecchio, Rachel M. Bittner, Aparna Kumar and Tillman Weyde. **Singing Voice Separation with Deep U-Net Convolutional Networks. ISMIR (2017).**


## Contributors

[Emoji key](https://allcontributors.org/docs/en/emoji-key)

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center">
        <a href="https://github.com/karlosos"><img src="https://avatars.githubusercontent.com/u/3882385?v=4" width="100px;" alt=""/><br /><sub><b>Karol Działowski</b></sub></a><br />
        📖 💻 🔣
    </td>
    <td align="center">
        <a href="https://github.com/jigiciak"><img src="https://avatars.githubusercontent.com/u/23162840?v=4" width="100px;" alt=""/><br /><sub><b>Marcin Łukasik</b></sub></a><br />
        📖 💻 🔣
    </td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
