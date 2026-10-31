import unittest
from unittest.mock import patch

import numpy as np

import main


class SignalProcessingTests(unittest.TestCase):
    def test_one_detection_cannot_match_two_reference_beats(self):
        actual, predicted = main.peaks_to_binary_labels(
            np.array([100, 120]), np.array([110]), 200
        )
        self.assertEqual(int(actual.sum()), 2)
        self.assertEqual(int(predicted.sum()), 1)

    def test_filter_rejects_powerline_and_preserves_passband(self):
        fs = 360
        time = np.arange(fs * 4) / fs
        wave = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 50 * time)
        _, filtered = main.apply_filters(wave, fs)
        original = np.abs(np.fft.rfft(wave))
        result = np.abs(np.fft.rfft(filtered))
        self.assertLess(result[50 * 4] / original[50 * 4], 0.1)
        self.assertGreater(result[10 * 4] / original[10 * 4], 0.7)

    def test_loaded_segment_has_ten_seconds_at_360_hz(self):
        with patch.object(main, "electrocardiogram", return_value=np.arange(108000)):
            time, signal, fs = main.load_data()
        self.assertEqual(fs, 360)
        self.assertEqual(len(signal), 3600)
        self.assertEqual(len(time), len(signal))
        self.assertAlmostEqual(float(signal.mean()), 0)


if __name__ == "__main__":
    unittest.main()
