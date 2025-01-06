# Fine Tuned TTS English-Bangla
# Overview
This project provides a Text-to-Speech (TTS) solution for both English and Bangla languages using two different TTS models. It utilizes the TTS library for English and the Silero model for Bangla. The project also includes functionality to transliterate Bangla text from Bengali script to Roman script for TTS processing.
# Requirements
To run this code, you'll need to have the following packages installed: <br />
  ●	TTS <br />
  ●	silero <br />
  ●	aksharamukha <br />
  ●	pandas <br />
  ●	torch <br />
You can install the required packages using pip: <br />
bash <br />
Copy code <br />
pip install TTS silero aksharamukha pandas torch <br />
# Dataset
The code requires two datasets:
  1.	English Dataset: A CSV file (eng_dataset.csv) containing English sentences. The expected column name is text or sentences.
  2.	Bangla Dataset: A CSV file (bangla_dataset.csv) containing Bangla sentences. The expected column name is text or sentences.

# Code Explanation
# Import Libraries
import pandas as pd <br />
import torch <br />
from TTS.api import TTS <br />
from IPython.display import Audio <br />
from aksharamukha import transliterate <br />

# Load Datasets
The code loads the English and Bangla datasets, renaming the columns if necessary: <br />
python <br />
english_data = pd.read_csv('eng_dataset.csv') <br />
bangla_data = pd.read_csv('bangla_dataset.csv') <br />

# Initialize TTS Models
  ●	English TTS: Using the Tacotron2 model.<br />
  ●	Bangla TTS: Using the Silero model for Indic languages. <br />
python <br />
english_model_name = "tts_models/en/ljspeech/tacotron2-DDC" <br />
english_tts_model = TTS(model_name=english_model_name) <br />

# from silero import silero_tts
model, example_text = silero_tts(language='indic', speaker='v3_indic') <br />
model = model.to(torch.device('cpu'))  # Or 'cuda' if using a GPU 

# Generate Bangla Speech
A function is provided to generate Bangla speech from text. It transliterates Bangla text from Bengali script to Roman script before applying TTS. <br />
python
def generate_bangla_speech(text): <br />
    roman_text = transliterate.process('Bengali', 'ISO', text) <br />
    audio = model.apply_tts(text=roman_text, speaker='bengali_female') <br />
    return audio<br />

Example Usage
You can generate and play audio for both English and Bangla texts as follows:

# Generate English audio
english_text = english_data['text'][0] <br />
english_audio = english_tts_model.tts(english_text) <br />
display(Audio(english_audio, rate=22050)) <br />

# Generate Bangla audio
bangla_text = bangla_data['text'][0] <br />
bangla_audio = generate_bangla_speech(bangla_text) <br />
display(Audio(bangla_audio, rate=48000)) <br />

# Acknowledgements
●	TTS library contributors <br />
●	Silero TTS contributors <br />
●	Aksharamukha project for transliteration <br />
●	Pandas library for data handling

