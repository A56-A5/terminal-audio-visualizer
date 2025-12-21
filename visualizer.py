import numpy as np
import soundcard as sc
import time
import sys
import shutil
import warnings
from soundcard import SoundcardRuntimeWarning

warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
FPS = 30
DECAY = 0.85
BAR_CHAR = "█" 

speaker = sc.default_speaker()
mic = sc.get_microphone(speaker.name, include_loopback=True)

prev = None

def render_vertical(values, height):
    width = len(values)
    values = np.interp(values, (0, np.max(values) + 1e-6), (0, height))
    canvas = [[" "] * width for _ in range(height)]
    for x in range(width):
        h = int(values[x])
        for y in range(h):
            canvas[height - 1 - y][x] = BAR_CHAR
    return "\n".join("".join(row) for row in canvas)

with mic.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE) as recorder:
    while True:
        term_size = shutil.get_terminal_size()
        width = term_size.columns
        height = term_size.lines - 1 

        if prev is None or len(prev) != width:
            prev = np.zeros(width)

        data = recorder.record(numframes=BLOCK_SIZE)
        mono = data.mean(axis=1)

        fft = np.abs(np.fft.rfft(mono))
        fft_resampled = np.interp(
            np.linspace(0, len(fft) - 1, width),
            np.arange(len(fft)),
            fft
        )
        fft_resampled /= (np.max(fft_resampled) + 1e-6)

        fft_resampled = np.maximum(fft_resampled, prev * DECAY)
        prev = fft_resampled

        sys.stdout.write("\033[H\033[J")
        sys.stdout.write(render_vertical(fft_resampled, height))
        sys.stdout.flush()

        time.sleep(1 / FPS)
