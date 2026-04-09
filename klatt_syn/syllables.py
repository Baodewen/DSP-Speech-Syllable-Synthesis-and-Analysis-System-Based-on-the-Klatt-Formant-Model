from __future__ import annotations

import json
import math
import struct
import wave
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

from .klatt import FrameParms, GlottalSourceType, MainParms, generate_sound

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CROSSFADE_MS = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY_PATH = PROJECT_ROOT / "presets" / "syllables.json"

_FRAME_FIELD_NAMES = [field.name for field in fields(FrameParms)]
_LIST_FIELDS = {"oral_formant_freq", "oral_formant_bw", "oral_formant_db"}
_BOOL_FIELDS = {"cascade_enabled", "parallel_enabled"}


@dataclass(frozen=True)
class SyllableSegment:
    glottal_source_type: GlottalSourceType
    frames: list[FrameParms]


@dataclass(frozen=True)
class SyllablePreset:
    id: str
    label: str
    description: str
    segments: list[SyllableSegment]


@dataclass(frozen=True)
class SyllableLibrary:
    sample_rate: int
    default_crossfade_ms: int
    presets: list[SyllablePreset]

    def get_preset(self, preset_id: str) -> SyllablePreset:
        for preset in self.presets:
            if preset.id == preset_id:
                return preset
        raise KeyError(f"Unknown preset id: {preset_id}")


__all__ = [
    "DEFAULT_CROSSFADE_MS",
    "DEFAULT_LIBRARY_PATH",
    "DEFAULT_SAMPLE_RATE",
    "PROJECT_ROOT",
    "SyllableLibrary",
    "SyllablePreset",
    "SyllableSegment",
    "export_wav",
    "load_syllable_library",
    "render_sequence",
    "render_syllable",
]


def _decode_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Field '{field_name}' must be numeric.")
    if not math.isfinite(float(value)):
        raise ValueError(f"Field '{field_name}' must be finite.")
    return int(value)


def _decode_float(value: object, field_name: str) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Field '{field_name}' must be numeric or null.")
    if not math.isfinite(float(value)):
        raise ValueError(f"Field '{field_name}' must be finite.")
    return float(value)


def _decode_float_list(value: object, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"Field '{field_name}' must be a list.")
    return [_decode_float(item, field_name) for item in value]


def _parse_glottal_source_type(value: object) -> GlottalSourceType:
    if not isinstance(value, str):
        raise ValueError("Field 'glottal_source_type' must be a string.")
    try:
        return GlottalSourceType(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported glottal source type: {value}") from exc


def _parse_frame(raw: object) -> FrameParms:
    if not isinstance(raw, dict):
        raise ValueError("Each frame must be an object.")
    missing = [name for name in _FRAME_FIELD_NAMES if name not in raw]
    if missing:
        raise ValueError(f"Frame is missing required fields: {', '.join(missing)}")

    kwargs: dict[str, object] = {}
    for field_name in _FRAME_FIELD_NAMES:
        value = raw[field_name]
        if field_name in _LIST_FIELDS:
            kwargs[field_name] = _decode_float_list(value, field_name)
        elif field_name in _BOOL_FIELDS:
            if not isinstance(value, bool):
                raise ValueError(f"Field '{field_name}' must be a boolean.")
            kwargs[field_name] = value
        else:
            kwargs[field_name] = _decode_float(value, field_name)
    return FrameParms(**kwargs)


def _parse_segment(raw: object) -> SyllableSegment:
    if not isinstance(raw, dict):
        raise ValueError("Each segment must be an object.")
    if "glottal_source_type" not in raw or "frames" not in raw:
        raise ValueError("Each segment requires 'glottal_source_type' and 'frames'.")
    frames_raw = raw["frames"]
    if not isinstance(frames_raw, list) or not frames_raw:
        raise ValueError("Each segment must contain a non-empty 'frames' list.")
    return SyllableSegment(
        glottal_source_type=_parse_glottal_source_type(raw["glottal_source_type"]),
        frames=[_parse_frame(frame) for frame in frames_raw],
    )


def _parse_preset(raw: object) -> SyllablePreset:
    if not isinstance(raw, dict):
        raise ValueError("Each preset must be an object.")
    required = ["id", "label", "description", "segments"]
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"Preset is missing required fields: {', '.join(missing)}")
    if not all(isinstance(raw[name], str) and raw[name].strip() for name in ["id", "label", "description"]):
        raise ValueError("Preset id, label, and description must be non-empty strings.")
    segments_raw = raw["segments"]
    if not isinstance(segments_raw, list) or not segments_raw:
        raise ValueError(f"Preset '{raw['id']}' must have a non-empty 'segments' list.")
    return SyllablePreset(
        id=raw["id"],
        label=raw["label"],
        description=raw["description"],
        segments=[_parse_segment(segment) for segment in segments_raw],
    )


def load_syllable_library(path: str | Path) -> SyllableLibrary:
    library_path = Path(path)
    raw = json.loads(library_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("The syllable library must be a JSON object.")
    for field_name in ("sample_rate", "default_crossfade_ms", "presets"):
        if field_name not in raw:
            raise ValueError(f"Library is missing required field '{field_name}'.")

    sample_rate = _decode_int(raw["sample_rate"], "sample_rate")
    default_crossfade_ms = _decode_int(raw["default_crossfade_ms"], "default_crossfade_ms")
    presets_raw = raw["presets"]
    if not isinstance(presets_raw, list) or not presets_raw:
        raise ValueError("The syllable library must contain a non-empty 'presets' list.")

    presets = [_parse_preset(preset) for preset in presets_raw]
    seen_ids: set[str] = set()
    for preset in presets:
        if preset.id in seen_ids:
            raise ValueError(f"Duplicate preset id: {preset.id}")
        seen_ids.add(preset.id)

    return SyllableLibrary(
        sample_rate=sample_rate,
        default_crossfade_ms=default_crossfade_ms,
        presets=presets,
    )


def render_syllable(preset: SyllablePreset, sample_rate: int | None = None) -> list[float]:
    actual_sample_rate = sample_rate or DEFAULT_SAMPLE_RATE
    out: list[float] = []
    for segment in preset.segments:
        main_parms = MainParms(
            sample_rate=actual_sample_rate,
            glottal_source_type=segment.glottal_source_type,
        )
        out.extend(generate_sound(main_parms, segment.frames))
    return out


def _raised_cosine_crossfade(left: list[float], right: list[float], overlap: int) -> list[float]:
    if not left:
        return list(right)
    if not right or overlap <= 0:
        return left + list(right)

    actual_overlap = min(overlap, len(left), len(right))
    if actual_overlap <= 0:
        return left + list(right)

    merged = list(left[:-actual_overlap])
    for index in range(actual_overlap):
        phase = index / (actual_overlap - 1) if actual_overlap > 1 else 1.0
        fade_in = 0.5 - 0.5 * math.cos(math.pi * phase)
        fade_out = 1.0 - fade_in
        merged.append(left[-actual_overlap + index] * fade_out + right[index] * fade_in)
    merged.extend(right[actual_overlap:])
    return merged


def render_sequence(
    presets: Iterable[SyllablePreset],
    sample_rate: int | None = None,
    crossfade_ms: int | None = None,
) -> list[float]:
    actual_sample_rate = sample_rate or DEFAULT_SAMPLE_RATE
    actual_crossfade_ms = DEFAULT_CROSSFADE_MS if crossfade_ms is None else int(crossfade_ms)
    overlap = max(0, round(actual_sample_rate * actual_crossfade_ms / 1000.0))

    out: list[float] = []
    for preset in presets:
        syllable_samples = render_syllable(preset, actual_sample_rate)
        out = _raised_cosine_crossfade(out, syllable_samples, overlap)
    return out


def export_wav(path: str | Path, samples: list[float], sample_rate: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = [max(-1.0, min(1.0, sample)) for sample in samples]
    frames = b"".join(struct.pack("<h", int(sample * 32767.0)) for sample in clipped)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

