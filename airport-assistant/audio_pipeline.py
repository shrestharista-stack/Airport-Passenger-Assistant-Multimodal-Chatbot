# audio preprocessing- preparing audio to provide them to Whisper in the expected format.

import librosa
import torch


#defining function to load audio file from the path. The desired sampling rate(sr) is 16000 samples per second
def load_audio(path, sr=16000):
    # loading audio using librosa which converts audio into waveforms (numbers) and returns it along with sampling rate.
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

#defining a function to trim the start and end of session with no audio. The threshold for the audio is 25 decibels.
def trim_audio(y, top_db=25):
    y_trim, _ = librosa.effects.trim(y, top_db=top_db)
    #checking if the trimmed audio is empty, if yes then return to orgigial audio
    if len(y_trim) == 0:
        return y
    return y_trim

#defining a function to extract MFCC features. (MFCC are the numerical features to represent characteristics of audio)
def get_mfcc(y, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

#Creating main class to handle audio analysis
class WhisperTranscriber:
    #Set up the processor and choose the hardware
    def __init__(self, model_name="openai/whisper-tiny", device='cuda'):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.device = device
        print("Loading Whisper:", model_name, "on", device)
        #Load the tool that prepares audio for the model
        self.processor = WhisperProcessor.from_pretrained(model_name)
        #Load the actual AI model that understands speech
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        # Move the model to the GPU memory
        self.model.to(device)
        # Set the model to 'evaluation mode'
        self.model.eval()

    #defining function to transcribe the audio words into text
    def transcribe(self, path):
        y, sr = load_audio(path, sr=16000)
        y = trim_audio(y)
        #preparing the audio for the model at a standard speed
        inputs = self.processor(y, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        #model is not learning here
        with torch.no_grad():
            #generating numerical codes of the audio
            predicted_ids = self.model.generate(input_features)
            #converting back codes into human language
            text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return text.strip()
