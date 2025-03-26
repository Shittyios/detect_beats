import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame
import threading

def play_audio(audio_path, volume=0.3):
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.set_volume(volume)
    pygame.mixer.music.play()

def detect_and_visualize_beats(audio_path):
    pygame.mixer.init()
    y, sr = librosa.load(audio_path, sr=None)
    print(f"Audio loaded. Sampling rate: {sr}, Total samples: {len(y)}")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    fig, ax = plt.subplots(figsize=(12, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.6, color='b', ax=ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    for onset in onsets:
        ax.axvline(x=onset, color='g', alpha=0.5, linestyle='--')

    playhead_line, = ax.plot([0, 0], [-1, 1], color="green", linewidth=2, label="Playhead")
    ax.legend()
    last_onset_idx = 0
    def update_playhead(frame):
        nonlocal last_onset_idx
        current_time = pygame.mixer.music.get_pos() / 1000
        playhead_line.set_xdata([current_time, current_time])
        if last_onset_idx < len(onsets) and current_time >= onsets[last_onset_idx] - 0.1 :
            print(f"{onsets[last_onset_idx]:.2f},", end=" ", flush=True)
            last_onset_idx += 1

        return playhead_line,

    duration = len(y) / sr
    num_frames = int(duration * sr / 512)
    ani = FuncAnimation(fig, update_playhead, frames=num_frames, interval=1000 / sr, blit=True)


    audio_thread = threading.Thread(target=play_audio, args=(audio_path, 0.3))  #volume to 1 is usually loud
    audio_thread.start()
    plt.show()
    audio_thread.join()


audio_file = "ratatouile.wav"
detect_and_visualize_beats(audio_file)
