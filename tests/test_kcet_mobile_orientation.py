import os
import cv2
import numpy as np
import json
from pathlib import Path
from scanner import load_image, process_omr
from omr_preprocess.registration_align import detect_registration_blocks


def test_4_way_rotation_registration_detection():
    """Verify registration block detection succeeds on 0°, 90°, 180°, and 270° rotated inputs."""
    h_port, w_port = 1400, 1000
    sheet = np.full((h_port, w_port, 3), 245, dtype=np.uint8)
    
    # Place registration blocks near corners (TL, TR, BR, BL)
    cv2.rectangle(sheet, (80, 80), (160, 160), (10, 10, 10), -1)    # TL
    cv2.rectangle(sheet, (840, 80), (920, 160), (10, 10, 10), -1)   # TR
    cv2.rectangle(sheet, (840, 1240), (920, 1320), (10, 10, 10), -1) # BR
    cv2.rectangle(sheet, (80, 1240), (160, 1320), (10, 10, 10), -1)  # BL

    # Test 0 degrees (natural)
    markers_0, _ = detect_registration_blocks(sheet)
    assert len(markers_0) == 4

    # Test 90° CW
    sheet_90 = cv2.rotate(sheet, cv2.ROTATE_90_CLOCKWISE)
    markers_90, _ = detect_registration_blocks(sheet_90)
    assert len(markers_90) == 4

    # Test 180°
    sheet_180 = cv2.rotate(sheet, cv2.ROTATE_180)
    markers_180, _ = detect_registration_blocks(sheet_180)
    assert len(markers_180) == 4

    # Test 270° (90° CCW)
    sheet_270 = cv2.rotate(sheet, cv2.ROTATE_90_COUNTERCLOCKWISE)
    markers_270, _ = detect_registration_blocks(sheet_270)
    assert len(markers_270) == 4
