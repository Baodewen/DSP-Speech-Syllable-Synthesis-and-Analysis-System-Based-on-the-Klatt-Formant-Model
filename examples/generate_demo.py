from __future__ import annotations

import math
import pathlib
import struct
import sys
import wave

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from klatt_syn import GlottalSourceType, MainParms, demo_frame_params, generate_sound


def fade_audio_signal_in_place(samples: list[float], fade_margin: int) -> None:
    d = min(len(samples), 2 * fade_margin)
    if d <= 1:
        return
    for i in range(d // 2):
        x = i / d
        w = 0.5 - 0.5 * math.cos(2 * math.pi * x)
        samples[i] *= w
        samples[len(samples) - 1 - i] *= w


def write_wav_pcm16(path: pathlib.Path, samples: list[float], sample_rate: int) -> None:
    clipped = [max(-1.0, min(1.0, sample)) for sample in samples]
    frames = b"".join(struct.pack("<h", int(sample * 32767.0)) for sample in clipped)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)


def main() -> None:
    sample_rate = 44100
    main_parms = MainParms(
        sample_rate=sample_rate,
        glottal_source_type=GlottalSourceType.IMPULSIVE,
    )
    signal_samples = generate_sound(main_parms, [demo_frame_params])
    fade_audio_signal_in_place(signal_samples, int(0.05 * sample_rate))
    output_file = pathlib.Path(__file__).with_name("example.wav")
    write_wav_pcm16(output_file, signal_samples, sample_rate)
    print(f'Audio data written to "{output_file}".')


if __name__ == "__main__":
    main()
