import cv2
import numpy as np
import glob
import os

folder = "data/test/yellow"  # change to your folder

for path in glob.glob(os.path.join(folder, "*.png")):
    img = cv2.imread(path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mask = v > 20  # optional

    print("----", path)
    print("H range:", h.min(), h.max())
    print("S range:", s.min(), s.max())
    print("V range:", v.min(), v.max())
