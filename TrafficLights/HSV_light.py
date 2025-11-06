import numpy as np
import cv2

def hsv_fallback(pil_crop):
    bgr = cv2.cvtColor(np.array(pil_crop), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)

    # score by color ranges (tune thresholds if needed)
    red_mask1 = ((H<10) & (S>80) & (V>120))
    red_mask2 = ((H>170) & (S>80) & (V>120))
    yellow_mask = ((H>=15)&(H<=35) & (S>80) & (V>120))
    green_mask = ((H>=40)&(H<=85) & (S>80) & (V>120))

    scores = {
        "red":    red_mask1.sum() + red_mask2.sum(),
        "yellow": yellow_mask.sum(),
        "green":  green_mask.sum()
    }
    return max(scores, key=scores.get)