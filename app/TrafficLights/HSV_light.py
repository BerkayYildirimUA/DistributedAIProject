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





import numpy as np
import cv2

def hsv_color_from_bgr_crop(bgr_crop):
    """Return ('red'|'yellow'|'green'|'unknown', confidence 0..1) for a BGR crop."""
    if bgr_crop is None or bgr_crop.size == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # OpenCV ranges: H in [0,179], S,V in [0,255]
    v_mean = float(np.mean(V))
    s_thr = 60 if v_mean < 100 else 80
    v_thr = 80 if v_mean < 100 else 120

    red_mask  = ((H < 10) | (H > 170)) & (S > s_thr) & (V > v_thr)
    yellow_m  = (H >= 15) & (H <= 35)  & (S > s_thr) & (V > v_thr)
    green_m   = (H >= 40) & (H <= 85)  & (S > s_thr) & (V > v_thr)

    # Simple noise suppression: keep median-blurred masks
    def score(mask):
        if mask.sum() == 0: return 0
        m = (mask.astype(np.uint8) * 255)
        m = cv2.medianBlur(m, 3)
        return int((m > 0).sum())

    scores = {
        "red":    score(red_mask),
        "yellow": score(yellow_m),
        "green":  score(green_m),
    }

    label = max(scores, key=scores.get)
    bright = (V > v_thr)
    denom = int(bright.sum()) or 1
    conf = float(scores[label]) / denom
    if scores[label] == 0:
        return "unknown", 0.0
    return label, conf
