import unittest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import cv2
from scanner import detect_jee_series, load_template

class TestJeeSeries(unittest.TestCase):

    def test_detect_jee_series_auto_coordinates(self):
        # Load jee template
        template_path = os.path.join(
            os.path.dirname(__file__), "..", "templates", "jee.json"
        )
        template = load_template(template_path)
        
        # Create a synthetic grayscale image (1054 x 1492)
        h, w = 1492, 1054
        gray = np.full((h, w), 240, dtype=np.uint8)
        
        # Fill bubble for series "A" at (272, 157)
        cv2.circle(gray, (272, 157), 8, (20, 20, 20), -1)
        
        result = detect_jee_series(gray, template)
        self.assertIsNotNone(result)
        self.assertEqual(result["value"], "A")

if __name__ == "__main__":
    unittest.main()
