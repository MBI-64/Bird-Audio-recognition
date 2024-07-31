from tkinter import *
from tkinter import filedialog
import pygame
import time
import tkinter.ttk as ttk
import tensorflow as tf 
from tensorflow.keras.models import model_from_json
from itertools import groupby
import librosa
import os

songs_with_path = {}

class Birds:
    pygame.mixer.init()

    global stopped
    stopped = False

    global paused
    paused = False

    def add_song(self):
        global song_path
    
        song_path = filedialog.askopenfilename(initialdir='newdata', title='Choose A Song', filetypes=(("mp3 Files", "*.mp3" ), ("All", "*")))
        song = song_path.split('/')[-1]
        song = song.replace('.wav', '')
        songs_with_path[song] = song_path
        playlist_box.insert(END, song)

    def play(self):    
        global status_bar
        global song_slider
        global stopped
        global my_sound

        status_bar.config(text='00:00  /00:00')
        song_slider.config(value=0)
        stopped = False

        songs  = str(playlist_box.get(ACTIVE))
        song = songs_with_path[songs]
        my_sound = pygame.mixer.Sound(song)
        my_sound.play(loops=0)

        self.play_time()
      
  
    def stop(self):
        global status_bar
        global song_slider
        global stopped
        global my_sound
        
        pygame.mixer.Sound.stop(my_sound)
        playlist_box.selection_clear(ACTIVE)

        status_bar.config(text='00:00  /00:00')

        song_slider.config(value=0)
        stopped = True

    def display (self):
        global song_path

        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5")
        print("Model Loaded")

        def load_mp3(filename):
            filename = filename.numpy().decode('utf-8')  # Decode bytes to string
            wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
            return wav

        def load_mp3_wrapper(filename):
            wav = tf.py_function(load_mp3, [filename], tf.float32)
            wav.set_shape([None])
            return wav

        # Define preprocess_mp3 function
        def preprocess_mp3(sample, index):
            sample = sample[0]  # Extract the sample from the batch
            zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
            wav = tf.concat([zero_padding, sample], 0)
            spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
            spectrogram = tf.abs(spectrogram)
            spectrogram = tf.expand_dims(spectrogram, axis=2)
            return spectrogram

        mp3 = os.path.join(song_path)
        wav = load_mp3_wrapper(tf.constant(mp3))

        audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
        audio_slices = audio_slices.map(preprocess_mp3)
        audio_slices = audio_slices.batch(16) 

        yhat = loaded_model.predict(audio_slices)
        yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

        yhat = [key for key, group in groupby(yhat)]
        calls = tf.math.reduce_sum(yhat).numpy()

        print("Number of calls in the audio is: ", calls)

        total.configure(text=calls)

    def play_time(self):
        global stopped
        global playlist_box
        global song_slider
        global song_length
        global status_bar
        global my_sound

        if stopped:
            return

        else:
            current_time = pygame.mixer.music.get_pos() / 1000

            converted_time = time.strftime('%M:%S', time.gmtime(current_time))
            current_song = playlist_box.get(ACTIVE)
            song = songs_with_path[current_song]
            song_length = my_sound.get_length()
            
            converted_song_length = time.strftime('%M:%S', time.gmtime(song_length))

            if int(song_slider.get()) == int(song_length):
                self.display()
                self.stop()
            
            elif paused:
                pass

            else:
                next_time = int(song_slider.get()) + 1

                song_slider.config(value=next_time, to=song_length )
                converted_time = time.strftime('%M:%S', time.gmtime(int(song_slider.get())))

                status_bar.config(text=f'{converted_time}  /{converted_song_length}')

            if current_time >= 0:
                status_bar.config(text=f'{converted_time}  /{converted_song_length}')

            status_bar.after(1000, self.play_time)


    def slide(self, x):
        global playlist_box
        global song_slider
        
        songs  = str(playlist_box.get(ACTIVE))
        song = songs_with_path[songs]
        pygame.mixer.Sound(song)
        pygame.mixer.Sound.play(loops=0, start=song_slider.get())


    def main(self):

        global playlist_box
        tkinter_instance = Tk()
        tkinter_instance.title('Audio Classification')
        tkinter_instance.geometry("500x400")

        main_frame = Frame(tkinter_instance)

        main_frame.pack(pady=20)

        playlist_box = Listbox(main_frame, bg='black', fg='white', width=50, selectbackground='green', selectforeground='black')
        playlist_box.grid(row=0, column=0)

        global total

        total = Label(main_frame, bg='black', fg='white', width=10)

        total.grid(row=0, column=1)
        
        play_btn_img = PhotoImage(file='images/play.png')
        stop_btn_img = PhotoImage(file='images/stop.png')
        browse_btn_img = PhotoImage(file='images/browse.png')

        button_controller = Frame(main_frame)

        button_controller.grid(row=1, column=0, pady=20)
     
        stop_button = Button(button_controller, image=stop_btn_img, borderwidth=0, command=self.stop)
        browse_button = Button(button_controller, image=browse_btn_img, borderwidth=0, command=self.add_song)
        play_button = Button(button_controller, image=play_btn_img, borderwidth=0, command=self.play)

   
        play_button.grid(row=0, column=1, padx=5)
        stop_button.grid(row=0, column=3, padx=5)
        browse_button.grid(row=0, column=4, padx=5)

        my_menu = Menu(tkinter_instance)
        tkinter_instance.config(menu=my_menu)

        global status_bar

        status_bar = Label(tkinter_instance, text='00:00  /00:00', bd=1, relief=GROOVE, anchor=E)
        status_bar.pack(fill=X, side=BOTTOM, pady=2)

        global song_slider

        song_slider = ttk.Scale(main_frame, from_=0, to=100, orient=HORIZONTAL, length=360, value=0, command=self.slide)
        song_slider.grid(row=2, column=0, pady=0)

        tkinter_instance.mainloop()

bird = Birds()    

bird.main()
