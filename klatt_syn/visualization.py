from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from .syllables import DEFAULT_CROSSFADE_MS, DEFAULT_SAMPLE_RATE, SyllablePreset, render_sequence, render_syllable

matplotlib.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
    }
)

FORMANT_COLORS = ["#8c1d18", "#1f5aa6", "#177e89"]
SYLLABLE_COLORS = ["#dbe9f6", "#fbe4d8", "#e2f0cb", "#f5d7ef", "#fbe9a7", "#d7f0f3"]
SOURCE_COLORS = {
    "impulsive": "#355070",
    "natural": "#6d597a",
    "noise": "#bc4749",
}


@dataclass(frozen=True)
class SequencePlotData:
    samples: np.ndarray
    sample_rate: int
    syllable_spans: list[tuple[float, float, str]]
    segment_spans: list[tuple[float, float, str, str]]
    crossfade_spans: list[tuple[float, float]]
    frame_centers: np.ndarray
    f0_values: np.ndarray
    formant_values: np.ndarray

    @property
    def duration(self) -> float:
        return 0.0 if self.samples.size == 0 else self.samples.size / self.sample_rate


__all__ = [
    "SequencePlotData",
    "build_sequence_plot_data",
    "render_formant_figure",
    "render_spectrogram_figure",
    "render_timeline_figure",
    "render_waveform_figure",
]


def build_sequence_plot_data(
    presets: Iterable[SyllablePreset],
    sample_rate: int | None = None,
    crossfade_ms: int | None = None,
) -> SequencePlotData:
    preset_list = list(presets)
    actual_sample_rate = sample_rate or DEFAULT_SAMPLE_RATE
    actual_crossfade_ms = DEFAULT_CROSSFADE_MS if crossfade_ms is None else int(crossfade_ms)
    requested_overlap = max(0, round(actual_sample_rate * actual_crossfade_ms / 1000.0))

    samples = np.asarray(
        render_sequence(preset_list, sample_rate=actual_sample_rate, crossfade_ms=actual_crossfade_ms),
        dtype=float,
    )

    syllable_spans: list[tuple[float, float, str]] = []
    segment_spans: list[tuple[float, float, str, str]] = []
    crossfade_spans: list[tuple[float, float]] = []
    frame_centers: list[float] = []
    f0_values: list[float] = []
    formant_values: list[list[float]] = []

    current_total = 0
    previous_length = 0
    for index, preset in enumerate(preset_list):
        syllable_samples = render_syllable(preset, actual_sample_rate)
        actual_overlap = 0 if index == 0 else min(requested_overlap, previous_length, len(syllable_samples))
        preset_start = 0 if index == 0 else current_total - actual_overlap
        if actual_overlap > 0:
            crossfade_spans.append(
                (preset_start / actual_sample_rate, (preset_start + actual_overlap) / actual_sample_rate)
            )

        local_cursor = 0
        for segment in preset.segments:
            segment_start = preset_start + local_cursor
            segment_length = 0
            for frame in segment.frames:
                frame_len = round(frame.duration * actual_sample_rate)
                frame_start = preset_start + local_cursor
                frame_end = frame_start + frame_len
                frame_centers.append((frame_start + frame_end) / 2 / actual_sample_rate)
                f0_values.append(frame.f0)
                padded_formants = list(frame.oral_formant_freq[:3])
                while len(padded_formants) < 3:
                    padded_formants.append(float("nan"))
                formant_values.append(padded_formants)
                local_cursor += frame_len
                segment_length += frame_len
            segment_spans.append(
                (
                    segment_start / actual_sample_rate,
                    (segment_start + segment_length) / actual_sample_rate,
                    segment.glottal_source_type.value,
                    preset.label,
                )
            )

        preset_end = preset_start + len(syllable_samples)
        syllable_spans.append((preset_start / actual_sample_rate, preset_end / actual_sample_rate, preset.label))
        current_total = preset_end
        previous_length = len(syllable_samples)

    return SequencePlotData(
        samples=samples,
        sample_rate=actual_sample_rate,
        syllable_spans=syllable_spans,
        segment_spans=segment_spans,
        crossfade_spans=crossfade_spans,
        frame_centers=np.asarray(frame_centers, dtype=float),
        f0_values=np.asarray(f0_values, dtype=float),
        formant_values=np.asarray(formant_values, dtype=float) if formant_values else np.empty((0, 3), dtype=float),
    )


def _style_empty_figure(fig: Figure, title: str, message: str) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.52, message, ha="center", va="center", fontsize=8)
    ax.set_title(title, pad=10)
    fig.tight_layout()


def _prepare_axes(fig: Figure, title: str):
    fig.clear()
    fig.set_dpi(300)
    fig.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.set_title(title, pad=8)
    return ax


def _apply_common_time_overlays(ax, data: SequencePlotData) -> None:
    for start, end in data.crossfade_spans:
        ax.axvspan(start, end, color="#f4a261", alpha=0.15, linewidth=0)
    for start, _, _ in data.syllable_spans[1:]:
        ax.axvline(start, color="#7a7a7a", linestyle="--", linewidth=0.7, alpha=0.7)


def _annotate_syllables(ax, data: SequencePlotData, y: float) -> None:
    for start, end, label in data.syllable_spans:
        ax.text((start + end) / 2, y, label, ha="center", va="bottom", fontsize=7)


def render_waveform_figure(fig: Figure, data: SequencePlotData | None, title: str = "Waveform") -> None:
    if data is None or data.samples.size == 0:
        _style_empty_figure(fig, title, "Select a syllable or sequence.")
        return
    ax = _prepare_axes(fig, title)
    time_axis = np.arange(data.samples.size) / data.sample_rate
    ax.plot(time_axis, data.samples, color="#1d3557", linewidth=0.8)
    _apply_common_time_overlays(ax, data)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.18, linewidth=0.4)
    if data.samples.size:
        peak = float(np.max(np.abs(data.samples)))
        ax.set_ylim(-max(0.1, peak * 1.15), max(0.1, peak * 1.15))
        _annotate_syllables(ax, data, ax.get_ylim()[1] * 0.84)
    fig.tight_layout()


def render_spectrogram_figure(fig: Figure, data: SequencePlotData | None, title: str = "Spectrogram") -> None:
    if data is None or data.samples.size < 32:
        _style_empty_figure(fig, title, "Not enough samples for spectrogram.")
        return
    ax = _prepare_axes(fig, title)
    ax.specgram(data.samples, NFFT=512, Fs=data.sample_rate, noverlap=384, cmap="magma")
    _apply_common_time_overlays(ax, data)
    ax.set_ylim(0, min(data.sample_rate / 2, 8000))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()


def render_formant_figure(fig: Figure, data: SequencePlotData | None, title: str = "Formant Trajectories") -> None:
    if data is None or data.formant_values.size == 0:
        _style_empty_figure(fig, title, "No formant data available.")
        return
    ax = _prepare_axes(fig, title)
    _apply_common_time_overlays(ax, data)
    labels = ["F1", "F2", "F3"]
    for index, label in enumerate(labels):
        ax.plot(
            data.frame_centers,
            data.formant_values[:, index],
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            color=FORMANT_COLORS[index],
            label=label,
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, max(4000, float(np.nanmax(data.formant_values[:, :3])) * 1.12))
    ax.grid(alpha=0.18, linewidth=0.4)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()


def render_timeline_figure(fig: Figure, data: SequencePlotData | None, title: str = "Sequence Timeline / F0") -> None:
    if data is None or data.samples.size == 0:
        _style_empty_figure(fig, title, "Select a syllable or sequence.")
        return
    ax = _prepare_axes(fig, title)
    for index, (start, end, label) in enumerate(data.syllable_spans):
        color = SYLLABLE_COLORS[index % len(SYLLABLE_COLORS)]
        ax.broken_barh([(start, end - start)], (58, 16), facecolors=color, edgecolors="#777777", linewidth=0.6)
        ax.text((start + end) / 2, 66, label, ha="center", va="center", fontsize=7)

    for start, end, source_type, _label in data.segment_spans:
        bar_color = SOURCE_COLORS.get(source_type, "#666666")
        ax.broken_barh([(start, end - start)], (26, 16), facecolors=bar_color, edgecolors="#666666", linewidth=0.5)
        ax.text((start + end) / 2, 34, source_type, ha="center", va="center", fontsize=6.5, color="white")

    for start, end in data.crossfade_spans:
        ax.axvspan(start, end, color="#f4a261", alpha=0.15, linewidth=0)

    ax2 = ax.twinx()
    if data.f0_values.size:
        ax2.plot(data.frame_centers, data.f0_values, color="#2a9d8f", linewidth=1.0, marker="o", markersize=2.5)
        ax2.set_ylim(0, max(300, float(np.max(data.f0_values)) * 1.2))
    else:
        ax2.set_ylim(0, 300)
    ax2.set_ylabel("F0 (Hz)")
    ax2.tick_params(labelsize=7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Structure")
    ax.set_ylim(0, 90)
    ax.set_yticks([34, 66])
    ax.set_yticklabels(["Source", "Syllable"])
    ax.grid(alpha=0.15, axis="x", linewidth=0.4)
    fig.tight_layout()
