import unittest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import cv2
from scanner import detect_corner_markers

class TestCornerDetection(unittest.TestCase):

    def test_detect_corner_markers_high_area_ratio(self):
        h, w = 1400, 1000
        img = np.full((h, w, 3), 100, dtype=np.uint8)
        cv2.rectangle(img, (30, 50), (970, 1350), (245, 245, 245), -1)
        
        template = {
            "sheet_width": 1054,
            "sheet_height": 1492,
        }
        
        corners = detect_corner_markers(img, template)
        self.assertIsNotNone(corners)
        self.assertEqual(len(corners), 4)

    def test_detect_corner_markers_full_frame(self):
        h, w = 1492, 1054
        img = np.full((h, w, 3), 245, dtype=np.uint8)
        cv2.putText(img, "OMR SHEET", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
        
        template = {
            "sheet_width": 1054,
            "sheet_height": 1492,
        }
        
        corners = detect_corner_markers(img, template)
        self.assertIsNotNone(corners)
        self.assertEqual(len(corners), 4)

    def test_detect_corner_markers_low_contrast_table(self):
        h, w = 1400, 1000
        img = np.full((h, w, 3), (210, 220, 225), dtype=np.uint8)
        cv2.rectangle(img, (50, 70), (950, 1330), (250, 250, 250), -1)
        
        template = {
            "sheet_width": 1054,
            "sheet_height": 1492,
        }
        
        corners = detect_corner_markers(img, template)
        self.assertIsNotNone(corners)
        self.assertEqual(len(corners), 4)

if __name__ == "__main__":
    unittest.main()

