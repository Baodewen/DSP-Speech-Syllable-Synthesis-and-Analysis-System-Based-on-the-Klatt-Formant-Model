from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from app.pyqt_panel import SyllablePanel
from klatt_syn import (
    DEFAULT_LIBRARY_PATH,
    build_sequence_plot_data,
    export_wav,
    load_syllable_library,
    render_sequence,
    render_syllable,
)


class SyllableLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = load_syllable_library(DEFAULT_LIBRARY_PATH)

    def test_load_library(self) -> None:
        self.assertEqual(self.library.sample_rate, 44100)
        self.assertEqual(self.library.default_crossfade_ms, 5)
        preset_ids = [preset.id for preset in self.library.presets]
        self.assertGreaterEqual(len(preset_ids), 17)
        for preset_id in ["a", "i", "u", "ai", "ha", "hi", "e", "o", "ae", "ei", "ou", "h", "s", "sh", "f", "m", "n"]:
            self.assertIn(preset_id, preset_ids)

    def test_render_hi_nonempty(self) -> None:
        hi = self.library.get_preset("hi")
        samples = render_syllable(hi, self.library.sample_rate)
        self.assertTrue(samples)
        self.assertGreater(max(abs(sample) for sample in samples), 0.01)

    def test_render_sequence_two_hi(self) -> None:
        hi = self.library.get_preset("hi")
        single = render_syllable(hi, self.library.sample_rate)
        doubled = render_sequence([hi, hi], self.library.sample_rate, self.library.default_crossfade_ms)
        self.assertGreater(len(doubled), len(single))
        self.assertLess(len(doubled), len(single) * 2)

    def test_build_sequence_plot_data(self) -> None:
        hi = self.library.get_preset("hi")
        data = build_sequence_plot_data([hi, hi], self.library.sample_rate, self.library.default_crossfade_ms)
        self.assertGreater(data.samples.size, 0)
        self.assertEqual(len(data.syllable_spans), 2)
        self.assertEqual(len(data.crossfade_spans), 1)
        self.assertEqual(data.formant_values.shape[1], 3)

    def test_export_wav(self) -> None:
        hi = self.library.get_preset("hi")
        samples = render_syllable(hi, self.library.sample_rate)
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "hi.wav"
            export_wav(wav_path, samples, self.library.sample_rate)
            self.assertTrue(wav_path.exists())
            with wave.open(str(wav_path), "rb") as wav_file:
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getframerate(), self.library.sample_rate)
                self.assertGreater(wav_file.getnframes(), 1000)

    def test_invalid_glottal_type_rejected(self) -> None:
        invalid_library = {
            "sample_rate": 44100,
            "default_crossfade_ms": 5,
            "presets": [
                {
                    "id": "bad",
                    "label": "bad",
                    "description": "bad preset",
                    "segments": [
                        {
                            "glottal_source_type": "unsupported",
                            "frames": [self._valid_frame()]
                        }
                    ]
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "invalid.json"
            json_path.write_text(json.dumps(invalid_library), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_syllable_library(json_path)

    def test_missing_frames_rejected(self) -> None:
        invalid_library = {
            "sample_rate": 44100,
            "default_crossfade_ms": 5,
            "presets": [
                {
                    "id": "bad",
                    "label": "bad",
                    "description": "bad preset",
                    "segments": [{"glottal_source_type": "impulsive"}]
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "missing_frames.json"
            json_path.write_text(json.dumps(invalid_library), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_syllable_library(json_path)

    def test_empty_presets_rejected(self) -> None:
        invalid_library = {
            "sample_rate": 44100,
            "default_crossfade_ms": 5,
            "presets": []
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "empty.json"
            json_path.write_text(json.dumps(invalid_library), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_syllable_library(json_path)

    @staticmethod
    def _valid_frame() -> dict[str, object]:
        return {
            "duration": 0.1,
            "f0": 200,
            "flutter_level": 0.2,
            "open_phase_ratio": 0.7,
            "breathiness_db": -30,
            "tilt_db": 4,
            "gain_db": None,
            "agc_rms_level": 0.18,
            "nasal_formant_freq": None,
            "nasal_formant_bw": None,
            "oral_formant_freq": [730, 1090, 2440, 3400, 4200, 5000],
            "oral_formant_bw": [80, 90, 120, 200, 300, 400],
            "cascade_enabled": True,
            "cascade_voicing_db": 0,
            "cascade_aspiration_db": -26,
            "cascade_aspiration_mod": 0.4,
            "nasal_antiformant_freq": None,
            "nasal_antiformant_bw": None,
            "parallel_enabled": False,
            "parallel_voicing_db": 0,
            "parallel_aspiration_db": -25,
            "parallel_aspiration_mod": 0.5,
            "frication_db": -30,
            "frication_mod": 0.5,
            "parallel_bypass_db": -99,
            "nasal_formant_db": None,
            "oral_formant_db": [0, -7, -16, -22, -30, -36],
        }


class SyllablePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.library = load_syllable_library(DEFAULT_LIBRARY_PATH)

    def setUp(self) -> None:
        self.panel = SyllablePanel(self.library, Path(DEFAULT_LIBRARY_PATH))
        self.app.processEvents()

    def tearDown(self) -> None:
        self.panel.close()
        self.app.processEvents()

    def test_sequence_operations(self) -> None:
        self._select_available_by_id("a")
        self.panel.add_selected_available_to_sequence()
        self._select_available_by_id("hi")
        self.panel.add_selected_available_to_sequence()
        self.assertEqual([preset.id for preset in self.panel.sequence_presets], ["a", "hi"])

        self.panel.sequence_list.setCurrentRow(1)
        self.panel.move_selected_sequence_item(-1)
        self.assertEqual([preset.id for preset in self.panel.sequence_presets], ["hi", "a"])

        self.panel.remove_selected_sequence_item()
        self.assertEqual([preset.id for preset in self.panel.sequence_presets], ["a"])

        self.panel.clear_sequence()
        self.assertEqual(self.panel.sequence_presets, [])
        self.assertEqual(self.panel.sequence_list.count(), 0)

    def test_report_figure_export(self) -> None:
        self._select_available_by_id("hi")
        self.panel.refresh_plot_views()
        self.assertIsNotNone(self.panel.current_plot_data)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_paths = self.panel.save_report_figures(temp_dir)
            self.assertEqual(len(output_paths), 4)
            for path in output_paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)

    def _select_available_by_id(self, preset_id: str) -> None:
        self.panel.filter_input.setText("")
        self.panel.refresh_available_list()
        for row in range(self.panel.available_list.count()):
            item = self.panel.available_list.item(row)
            if item.data(Qt.UserRole) == preset_id:
                self.panel.available_list.setCurrentRow(row)
                self.app.processEvents()
                return
        self.fail(f"Preset '{preset_id}' not found in available list")


if __name__ == "__main__":
    unittest.main()


