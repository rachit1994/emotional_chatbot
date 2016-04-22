#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import sys,os
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
from features import mfcc
import pickle
import heapq
import aiml
import pyttsx
import unirest
import json



def test_emo(test_file, gmms):
    """
        NOTE: Use only after training.
        Test a given file and predict an emotion for it.
    """
    rate, sig = wav.read(test_file)
    mfcc_feat = mfcc(sig, rate)
    pred = {}
    for emo in gmms:
        pred[emo] = gmms[emo].score(mfcc_feat)
    return emotions_nbest(pred, 2), pred


def emotions_nbest(d, n):
    """
        Utility function to return n best predictions for emotion.
    """
    return heapq.nlargest(n, d, key=lambda k: d[k])


def predict_emo(test_file, pickle_path = "./pickles"):
    """
        Description:
            Based on training or testing mode, takes appropriate path to predict emotions for input wav file.

        Params:
            * test_file (mandatory): Wav file for which emotion should be predicted.
            * pickle_path: Default value is same directory as the python file. Path to store pickle files for use later.
            * trng_path: Default is a folder called training in the enclosing directory. Folder containing training data.
            * training: Default is False. if made True, will start the training procedure before testing, else will used previously trained model to test.

        Return:
            A list of predicted emotion and next best predicted emotion.
    """
    gmms = pickle.load(open(pickle_path + "/gmmhmm.pkl", "rb"))
    predicted = test_emo(test_file, gmms)
    return predicted



def readAudio():
	r = sr.Recognizer()
	homedir=os.getcwd();
	with sr.Microphone() as source:
	    r.adjust_for_ambient_noise(source) # listen for 1 second to calibrate the energy threshold for ambient noise levels
	    print("Say something!")
	    audio = r.listen(source)
	with open("microphone-results.wav", "wb") as f:
	    f.write(audio.get_wav_data())
	(rate,sig) = wav.read("microphone-results.wav")

	return audio


def Audio_to_text(audio):
	text = sr.Recognizer().recognize_google(audio);
	return text

def Sentiment_from_text(text):
	response = unirest.post("https://community-sentiment.p.mashape.com/text/",
		headers={
		"X-Mashape-Key": "6kWx0pf49smshQK5IQRwCyi3Z2S7p1lGNvkjsnvVSC1E4CCCYk",
		"Content-Type": "application/x-www-form-urlencoded",
		"Accept": "application/json"
		},
		params={
		"txt": text
		}
		)
	return response

def chat_bot(text):
	bot = aiml.Kernel()
	bot.bootstrap(brainFile = "rachit.brn");
	return bot.respond(text)

def text_to_speech(text):
	engine = pyttsx.init()
	rate = engine.getProperty('rate')
	engine.setProperty('rate', rate-50)
	engine.say(text)
	engine.runAndWait()

def Input_audio_emotion():
	predicted, probs = predict_emo("microphone-results.wav")
    	return predicted[0]


if __name__ == "__main__":
	recorded_audio = readAudio()
	input = Audio_to_text(recorded_audio)
	emotions = Input_audio_emotion()
	reply = chat_bot(input)
	os.system('cls')
	print "\n\n U said: " + input + "\n\n"
	print "emotion from speech:  " + Input_audio_emotion() + "\n\n"
	print "sentiment analysis of input speech : \n " + Sentiment_from_text(input).raw_body + "\n\n"
	print "resulting emotion:  " + Input_audio_emotion() + "\n\n"
	print "My reply: " + reply + "\n\n" 
	print "sentiment analysis of input speech : \n" + Sentiment_from_text(reply).raw_body + "\n\n"
	text_to_speech(reply)
	

