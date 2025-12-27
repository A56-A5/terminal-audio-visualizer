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
GAP_CHAR = " "

speaker = sc.default_speaker()
mic = sc.get_microphone(speaker.name, include_loopback=True)

prev = None

def center_zigzag(values):
    n = len(values)
    out = np.zeros(n)
    center = n // 2
    left = center - 1
    right = center if n % 2 == 0 else center + 1

    out[center] = values[0]  
    idx = 1
    side = True  

    while idx < n:
        if side and right < n:
            out[right] = values[idx]
            right += 1
        elif not side and left >= 0:
            out[left] = values[idx]
            left -= 1
        idx += 1
        side = not side
    return out

def render_vertical(values, height):
    values = np.interp(values, (0, np.max(values) + 1e-6), (0, height))
    w = len(values)
    canvas = [[" "] * (w * 2 - 1) for _ in range(height)]
    for i, v in enumerate(values):
        h = int(v)
        x = i * 2
        for y in range(h):
            canvas[height - 1 - y][x] = BAR_CHAR
    return "\n".join("".join(row).rstrip() for row in canvas)


with mic.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE) as recorder:
    while True:
        term = shutil.get_terminal_size()
        width = term.columns
        height = term.lines - 1

        bars = (width + 1) // 2

        if prev is None or len(prev) != bars:
            prev = np.zeros(bars)

        data = recorder.record(numframes=BLOCK_SIZE)
        mono = data.mean(axis=1)

        fft = np.abs(np.fft.rfft(mono))
        fft[0] = 0

        spectrum = np.interp(
            np.linspace(0, len(fft) - 1, bars),
            np.arange(len(fft)),
            fft
        )

        spectrum = np.sqrt(spectrum)
        spectrum /= np.max(spectrum) + 1e-6

        spectrum = center_zigzag(spectrum)
        spectrum = np.maximum(spectrum, prev * DECAY)
        prev = spectrum

        sys.stdout.write("\033[H\033[J")
        sys.stdout.write(render_vertical(spectrum, height))
        sys.stdout.flush()

        time.sleep(1 / FPS)
