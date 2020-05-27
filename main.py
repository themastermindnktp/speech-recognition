import os
import threading
import tkinter
import pyaudio
import wave

from GMMHMM import gmmhmm


TITLE = "Word Reconigtion"
RESOLUTION = "300x150"
BUTTON_CONFIG = {
    'height': 1,
    'width': 15
}
LABEL_CONFIG = {
    'wraplength': 500
}

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
FRAME_PER_BUFFER = 1024

RECORDING_FILE = "temp.wav"


class Recorder:
    def __init__(self):
        self.start_button = tkinter.Button(
            root,
            text="Start Recording",
            command=self.start_recording,
            **BUTTON_CONFIG
        )
        self.start_button.pack()
        self.start_lock = False

        self.stop_button = tkinter.Button(
            root,
            text="Stop Recording",
            command=self.stop_recording,
            **BUTTON_CONFIG
        )
        self.stop_button.pack()
        self.stop_lock = True

        self.status = tkinter.Label(
            root,
            text="No recording"
        )
        self.status.pack()

        self.recognize_button = tkinter.Button(
            root,
            text="Recognize Word",
            command=self.recognize,
            **BUTTON_CONFIG
        )
        self.recognize_button.pack()
        self.recognize_lock = True

        self.is_recording = False


    def start_recording(self):
        if self.start_lock:
            return

        self.start_lock = True

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            frames_per_buffer=FRAME_PER_BUFFER,
            input=True
        )

        self.frames = []

        self.is_recording = True
        self.status.config(text="Recording")

        self.recognize_lock = True
        self.stop_lock = False

        thread = threading.Thread(target=self.record)
        thread.start()

    def stop_recording(self):
        if self.stop_lock:
            return

        self.stop_lock = True

        self.is_recording = False

        wave_file = wave.open("temp.wav", "wb")

        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(self.audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)

        wave_file.writeframes(b''.join(self.frames))
        wave_file.close()

        self.status.config(text="Recorded")

        self.recognize_lock = False
        self.start_lock = False

    def record(self):
        while (self.is_recording):
            data = self.stream.read(FRAME_PER_BUFFER)
            self.frames.append(data)

    def recognize(self):
        gmmhmm.trim_audio(RECORDING_FILE)
        mfcc = gmmhmm.get_mfcc(RECORDING_FILE)
        # mfcc = mfcc[0].reshape(-1, 1)
        probs = {} # save score(log_prob)
        labels = ["cua", "khong", "trong", "vietnam", "yte", "nguoi"]
        for i in labels:
            print(i)
            filename = ("GMMHMM/GMMHMM"+i)
            #print(filename)
            gh = gmmhmm.pickle.load(open(filename, 'rb'))
            probs[i] = gh.score(mfcc)
            print(probs[i])
        pred = max(probs.items(),key=gmmhmm.operator.itemgetter(1))[0]
        self.status.config(text=f"This is \"{pred}\"")


root = tkinter.Tk()
root.title(TITLE)
root.geometry(RESOLUTION)
app = Recorder()
root.mainloop()