import unittest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import cv2
from scanner import process_omr

class TestJeeAnswersScan(unittest.TestCase):

    def test_jee_full_scan_questions_detected(self):
        # Create a synthetic white canonical sheet 1054 x 1492
        h, w = 1492, 1054
        img = np.full((h, w, 3), 245, dtype=np.uint8)
        # Draw registration blocks defined in jee.json
        cv2.rectangle(img, (33, 37), (71, 75), (0, 0, 0), -1) # TL
        cv2.rectangle(img, (978, 37), (1016, 75), (0, 0, 0), -1) # TR
        cv2.rectangle(img, (33, 1371), (71, 1409), (0, 0, 0), -1) # BL
        cv2.rectangle(img, (978, 1371), (1016, 1409), (0, 0, 0), -1) # BR
        _, img_bytes = cv2.imencode('.png', img)
        
        template_path = os.path.join(
            os.path.dirname(__file__), "..", "templates", "jee.json"
        )
        
        processing = process_omr(img_bytes.tobytes(), template_path)
        answers = processing.get("answers", {})
        
        mcq = answers.get("mcq", {})
        numerical = answers.get("numerical", {})
        
        # 60 MCQ questions (20 physics + 20 chem + 20 math)
        self.assertEqual(len(mcq), 60)
        
        # 15 Numerical questions (5 physics + 5 chem + 5 math)
        self.assertEqual(len(numerical), 15)
        
        # Total 75 questions
        self.assertEqual(len(mcq) + len(numerical), 75)

if __name__ == "__main__":
    unittest.main()
