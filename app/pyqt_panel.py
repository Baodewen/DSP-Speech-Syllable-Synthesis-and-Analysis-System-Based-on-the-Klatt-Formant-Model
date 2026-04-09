from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QFont
from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from klatt_syn import DEFAULT_LIBRARY_PATH, SyllableLibrary, SyllablePreset, export_wav, load_syllable_library, render_sequence, render_syllable
from klatt_syn.visualization import (
    SequencePlotData,
    build_sequence_plot_data,
    render_formant_figure,
    render_spectrogram_figure,
    render_timeline_figure,
    render_waveform_figure,
)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 4.8, height: float = 3.2) -> None:
        self.figure = Figure(figsize=(width, height), dpi=300)
        super().__init__(self.figure)
        self.setMinimumSize(420, 280)


class SyllablePanel(QWidget):
    def __init__(self, library: SyllableLibrary, library_path: Path) -> None:
        super().__init__()
        self.library = library
        self.library_path = library_path
        self.sequence_presets: list[SyllablePreset] = []
        self.preview_counter = 0
        self.temp_dir = tempfile.TemporaryDirectory(prefix="klatt_syn_gui_")
        self.current_plot_data: SequencePlotData | None = None

        self.preview_effect = QSoundEffect(self)
        self.preview_effect.setLoopCount(1)
        self.preview_effect.setVolume(1.0)

        self.setWindowTitle("Klatt Syllable Panel")
        self.resize(1560, 920)
        self._apply_global_font()
        self._build_ui()
        self.refresh_available_list()
        self._set_status(
            f"Loaded {len(self.library.presets)} presets from {self.library_path.name} at {self.library.sample_rate} Hz."
        )
        self.update_button_states()
        self.refresh_plot_views()

    def _apply_global_font(self) -> None:
        self.setFont(QFont("Times New Roman", 9))

    def _build_ui(self) -> None:
        self.available_title = QLabel("Available syllables")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter syllables...")
        self.filter_input.textChanged.connect(self.refresh_available_list)

        self.available_list = QListWidget()
        self.available_list.itemDoubleClicked.connect(lambda _item: self.add_selected_available_to_sequence())
        self.available_list.itemSelectionChanged.connect(self._handle_available_selection_changed)

        self.preview_syllable_button = QPushButton("试听当前音节")
        self.preview_syllable_button.clicked.connect(self.play_selected_available)
        self.add_button = QPushButton("添加到序列 ->")
        self.add_button.clicked.connect(self.add_selected_available_to_sequence)

        available_layout = QVBoxLayout()
        available_layout.addWidget(self.available_title)
        available_layout.addWidget(self.filter_input)
        available_layout.addWidget(self.available_list)
        available_layout.addWidget(self.preview_syllable_button)
        available_layout.addWidget(self.add_button)

        self.sequence_title = QLabel("Sequence")
        self.sequence_list = QListWidget()
        self.sequence_list.itemSelectionChanged.connect(self.update_button_states)

        self.remove_button = QPushButton("删除")
        self.remove_button.clicked.connect(self.remove_selected_sequence_item)
        self.move_up_button = QPushButton("上移")
        self.move_up_button.clicked.connect(lambda: self.move_selected_sequence_item(-1))
        self.move_down_button = QPushButton("下移")
        self.move_down_button.clicked.connect(lambda: self.move_selected_sequence_item(1))
        self.clear_button = QPushButton("清空")
        self.clear_button.clicked.connect(self.clear_sequence)

        sequence_actions = QGridLayout()
        sequence_actions.addWidget(self.remove_button, 0, 0)
        sequence_actions.addWidget(self.move_up_button, 0, 1)
        sequence_actions.addWidget(self.move_down_button, 1, 0)
        sequence_actions.addWidget(self.clear_button, 1, 1)

        sequence_layout = QVBoxLayout()
        sequence_layout.addWidget(self.sequence_title)
        sequence_layout.addWidget(self.sequence_list)
        sequence_layout.addLayout(sequence_actions)

        self.preview_sequence_button = QPushButton("试听整段")
        self.preview_sequence_button.clicked.connect(self.play_sequence)
        self.export_button = QPushButton("导出 WAV")
        self.export_button.clicked.connect(self.export_sequence)
        self.export_report_button = QPushButton("导出图像集")
        self.export_report_button.clicked.connect(self.export_report_figures)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(60)
        self.report_hint_label = QLabel(
            "IEEE-style figures: separate plots, 300 DPI export, compact Times New Roman typography."
        )
        self.report_hint_label.setWordWrap(True)

        controls_layout = QVBoxLayout()
        controls_layout.addLayout(available_layout, 3)
        controls_layout.addSpacing(10)
        controls_layout.addLayout(sequence_layout, 3)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(self.preview_sequence_button)
        controls_layout.addWidget(self.export_button)
        controls_layout.addWidget(self.export_report_button)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QLabel("Status"))
        controls_layout.addWidget(self.status_label)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(self.report_hint_label)
        controls_layout.addStretch()

        controls_container = QWidget()
        controls_container.setLayout(controls_layout)
        controls_container.setMinimumWidth(360)
        controls_container.setMaximumWidth(420)

        self.wave_canvas = PlotCanvas()
        self.spectrogram_canvas = PlotCanvas()
        self.formant_canvas = PlotCanvas()
        self.timeline_canvas = PlotCanvas()

        plots_layout = QGridLayout()
        plots_layout.addWidget(self.wave_canvas, 0, 0)
        plots_layout.addWidget(self.spectrogram_canvas, 0, 1)
        plots_layout.addWidget(self.formant_canvas, 1, 0)
        plots_layout.addWidget(self.timeline_canvas, 1, 1)
        plots_layout.setHorizontalSpacing(12)
        plots_layout.setVerticalSpacing(12)

        plots_container = QWidget()
        plots_container.setLayout(plots_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(controls_container, 0)
        main_layout.addWidget(plots_container, 1)
        self.setLayout(main_layout)

    def refresh_available_list(self) -> None:
        filter_text = self.filter_input.text().strip().lower()
        selected_id = None
        current_item = self.available_list.currentItem()
        if current_item is not None:
            selected_id = current_item.data(Qt.UserRole)

        self.available_list.clear()
        for preset in self.library.presets:
            haystack = f"{preset.id} {preset.label} {preset.description}".lower()
            if filter_text and filter_text not in haystack:
                continue
            item = QListWidgetItem(f"{preset.label}  [{preset.id}]")
            item.setData(Qt.UserRole, preset.id)
            item.setToolTip(preset.description)
            self.available_list.addItem(item)
            if selected_id == preset.id:
                self.available_list.setCurrentItem(item)

        if self.available_list.currentRow() < 0 and self.available_list.count() > 0:
            self.available_list.setCurrentRow(0)
        self.update_button_states()
        self.refresh_plot_views()

    def add_selected_available_to_sequence(self) -> None:
        preset = self.selected_available_preset()
        if preset is None:
            return
        self.sequence_presets.append(preset)
        self._append_sequence_item(preset)
        self.sequence_list.setCurrentRow(self.sequence_list.count() - 1)
        self._set_status(f"Added '{preset.label}' to sequence.")
        self.update_button_states()
        self.refresh_plot_views()

    def remove_selected_sequence_item(self) -> None:
        row = self.sequence_list.currentRow()
        if row < 0:
            return
        removed = self.sequence_presets.pop(row)
        self.sequence_list.takeItem(row)
        if self.sequence_list.count() > 0:
            self.sequence_list.setCurrentRow(min(row, self.sequence_list.count() - 1))
        self._set_status(f"Removed '{removed.label}' from sequence.")
        self.update_button_states()
        self.refresh_plot_views()

    def move_selected_sequence_item(self, direction: int) -> None:
        row = self.sequence_list.currentRow()
        if row < 0:
            return
        target_row = row + direction
        if target_row < 0 or target_row >= len(self.sequence_presets):
            return
        self.sequence_presets[row], self.sequence_presets[target_row] = (
            self.sequence_presets[target_row],
            self.sequence_presets[row],
        )
        self._rebuild_sequence_list(target_row)
        self._set_status("Reordered sequence.")
        self.update_button_states()
        self.refresh_plot_views()

    def clear_sequence(self) -> None:
        self.sequence_presets.clear()
        self.sequence_list.clear()
        self._set_status("Cleared sequence.")
        self.update_button_states()
        self.refresh_plot_views()

    def play_selected_available(self) -> None:
        preset = self.selected_available_preset()
        if preset is None:
            QMessageBox.warning(self, "No selection", "Select a syllable to preview.")
            return
        samples = render_syllable(preset, self.library.sample_rate)
        self._preview_samples(samples, f"Previewed syllable '{preset.label}'.")

    def play_sequence(self) -> None:
        if not self.sequence_presets:
            QMessageBox.warning(self, "Empty sequence", "Add at least one syllable before previewing the sequence.")
            return
        samples = render_sequence(
            self.sequence_presets,
            sample_rate=self.library.sample_rate,
            crossfade_ms=self.library.default_crossfade_ms,
        )
        self._preview_samples(samples, f"Previewed sequence of {len(self.sequence_presets)} syllables.")

    def export_sequence(self) -> None:
        if not self.sequence_presets:
            QMessageBox.warning(self, "Empty sequence", "Add at least one syllable before exporting.")
            return
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export WAV",
            str(PROJECT_ROOT / "sequence.wav"),
            "WAV files (*.wav)",
        )
        if not output_path:
            return
        samples = render_sequence(
            self.sequence_presets,
            sample_rate=self.library.sample_rate,
            crossfade_ms=self.library.default_crossfade_ms,
        )
        export_wav(output_path, samples, self.library.sample_rate)
        self._set_status(f"Exported WAV to {output_path}.")

    def export_report_figures(self) -> None:
        if self.current_plot_data is None:
            QMessageBox.warning(self, "No visualization", "Select a syllable or build a sequence before exporting figures.")
            return
        output_dir = QFileDialog.getExistingDirectory(self, "Select export folder", str(PROJECT_ROOT))
        if not output_dir:
            return
        paths = self.save_report_figures(Path(output_dir))
        self._set_status(f"Exported {len(paths)} IEEE-style figures to {output_dir}.")

    def save_report_figures(self, output_dir: str | Path) -> list[Path]:
        if self.current_plot_data is None:
            raise ValueError("No plot data available.")
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        stem = self._current_report_stem()
        outputs = {
            "waveform": directory / f"{stem}_waveform.png",
            "spectrogram": directory / f"{stem}_spectrogram.png",
            "formants": directory / f"{stem}_formants.png",
            "timeline": directory / f"{stem}_timeline.png",
        }
        render_waveform_figure(self.wave_canvas.figure, self.current_plot_data, "Waveform")
        render_spectrogram_figure(self.spectrogram_canvas.figure, self.current_plot_data, "Spectrogram")
        render_formant_figure(self.formant_canvas.figure, self.current_plot_data, "Formant Trajectories")
        render_timeline_figure(self.timeline_canvas.figure, self.current_plot_data, "Sequence Timeline / F0")
        self.wave_canvas.figure.savefig(outputs["waveform"], dpi=300, bbox_inches="tight", facecolor="white")
        self.spectrogram_canvas.figure.savefig(outputs["spectrogram"], dpi=300, bbox_inches="tight", facecolor="white")
        self.formant_canvas.figure.savefig(outputs["formants"], dpi=300, bbox_inches="tight", facecolor="white")
        self.timeline_canvas.figure.savefig(outputs["timeline"], dpi=300, bbox_inches="tight", facecolor="white")
        self.wave_canvas.draw_idle()
        self.spectrogram_canvas.draw_idle()
        self.formant_canvas.draw_idle()
        self.timeline_canvas.draw_idle()
        return list(outputs.values())

    def selected_available_preset(self) -> SyllablePreset | None:
        item = self.available_list.currentItem()
        if item is None:
            return None
        return self.library.get_preset(item.data(Qt.UserRole))

    def current_plot_presets(self) -> list[SyllablePreset]:
        if self.sequence_presets:
            return list(self.sequence_presets)
        selected = self.selected_available_preset()
        return [selected] if selected is not None else []

    def refresh_plot_views(self) -> None:
        presets = self.current_plot_presets()
        self.current_plot_data = None
        if presets:
            self.current_plot_data = build_sequence_plot_data(
                presets,
                sample_rate=self.library.sample_rate,
                crossfade_ms=self.library.default_crossfade_ms,
            )

        render_waveform_figure(self.wave_canvas.figure, self.current_plot_data, "Waveform")
        render_spectrogram_figure(self.spectrogram_canvas.figure, self.current_plot_data, "Spectrogram")
        render_formant_figure(self.formant_canvas.figure, self.current_plot_data, "Formant Trajectories")
        render_timeline_figure(self.timeline_canvas.figure, self.current_plot_data, "Sequence Timeline / F0")
        self.wave_canvas.draw_idle()
        self.spectrogram_canvas.draw_idle()
        self.formant_canvas.draw_idle()
        self.timeline_canvas.draw_idle()
        self.update_button_states()

    def update_button_states(self) -> None:
        has_available = self.selected_available_preset() is not None
        has_sequence = bool(self.sequence_presets)
        has_report = self.current_plot_data is not None
        sequence_row = self.sequence_list.currentRow()

        self.preview_syllable_button.setEnabled(has_available)
        self.add_button.setEnabled(has_available)
        self.preview_sequence_button.setEnabled(has_sequence)
        self.export_button.setEnabled(has_sequence)
        self.export_report_button.setEnabled(has_report)
        self.remove_button.setEnabled(sequence_row >= 0)
        self.clear_button.setEnabled(has_sequence)
        self.move_up_button.setEnabled(sequence_row > 0)
        self.move_down_button.setEnabled(0 <= sequence_row < len(self.sequence_presets) - 1)

    def _append_sequence_item(self, preset: SyllablePreset) -> None:
        item = QListWidgetItem(f"{preset.label}  [{preset.id}]")
        item.setData(Qt.UserRole, preset.id)
        item.setToolTip(preset.description)
        self.sequence_list.addItem(item)

    def _rebuild_sequence_list(self, selected_row: int | None = None) -> None:
        self.sequence_list.clear()
        for preset in self.sequence_presets:
            self._append_sequence_item(preset)
        if selected_row is not None and self.sequence_list.count() > 0:
            self.sequence_list.setCurrentRow(selected_row)

    def _preview_samples(self, samples: list[float], status_text: str) -> None:
        if not samples:
            QMessageBox.warning(self, "No audio", "The selected syllable produced no samples.")
            return
        self.preview_effect.stop()
        preview_path = self._next_preview_path()
        export_wav(preview_path, samples, self.library.sample_rate)
        self.preview_effect.setSource(QUrl())
        self.preview_effect.setSource(QUrl.fromLocalFile(str(preview_path)))
        self.preview_effect.play()
        self._set_status(status_text)

    def _next_preview_path(self) -> Path:
        self.preview_counter += 1
        return Path(self.temp_dir.name) / f"preview_{self.preview_counter:04d}.wav"

    def _handle_available_selection_changed(self) -> None:
        self.update_button_states()
        self.refresh_plot_views()

    def _current_report_stem(self) -> str:
        if self.sequence_presets:
            return "sequence"
        selected = self.selected_available_preset()
        return selected.id if selected is not None else "klatt"

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.preview_effect.stop()
        self.temp_dir.cleanup()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv
    library_path = Path(args[1]) if len(args) > 1 else DEFAULT_LIBRARY_PATH
    app = QApplication(args)
    app.setApplicationName("Klatt Syllable Panel")
    app.setFont(QFont("Times New Roman", 9))

    try:
        library = load_syllable_library(library_path)
    except Exception as exc:
        QMessageBox.critical(None, "Failed to load library", str(exc))
        return 1

    panel = SyllablePanel(library, library_path)
    panel.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
