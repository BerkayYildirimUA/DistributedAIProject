class LaneDetection:
    def __init__(self):
        print("Lane detection created")
    
    def get_perspective_matrices(self, img):
        # Simulates a bird-eye view of the road from the front camera.
        # This performs a transformation that changes the traffic lines from the perspective of the car (1) to a birds-eye view (2)
        #
        #     \         /                   |       |         (1) is the src attribute, describing a trapezoid shape
        #      \       /        -->         |       |         (2) is the dst attribute, describing a rectangular shape      
        #       \_____/                     |_______|             ==> Allows us to more easily detect straight lines
        #         (1)                          (2)
        #
        # Note that this does not imply the use of an extra camera that provides a top-down view, it transforms the image collected by the 
        # front camera to hopefully be an accurate representation of a birds-eye view.

        src = np.float32([(200, 600), (200, 0), (630, 0), (630, 600)])
        h, w = img.shape[:2]
        img_size=(w, h)
        dst = np.float32([(0, h), (0, 0), (w, 0), (w, h)])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        return img

    def extract_features(self, img, nwindows):
        # Identify white pixels 
        window_height = int(img.shape[0] // nwindows)
        nonzero = img.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])
        return nonzerox, nonzeroy, window_height

    def hist(self, img):
        full_image = img[:,:]
        return np.sum(full_image, axis=0)

    def pixels_in_window(self, center, margin, height, nonzerox, nonzeroy):
        topleft = (center[0] - margin, center[1] - height//2)
        bottomright = (center[0] + margin, center[1] + height//2)
        condx = (topleft[0] <= nonzerox) & (nonzerox <= bottomright[0])
        condy = (topleft[1] <= nonzeroy) & (nonzeroy <= bottomright[1])
        return nonzerox[condx & condy], nonzeroy[condx & condy]

    def find_lane_pixels(self, img, nwindows, margin, minpix,nonzerox, nonzeroy, window_height):
        out_img = np.dstack((img, img, img))
        histogram = self.hist(img)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + window_height // 2
        leftx, lefty, rightx, righty = [], [], [], []

        for _ in range(nwindows):
            y_current -= window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)
            good_left_x, good_left_y = self.pixels_in_window(center_left, margin, window_height, nonzerox, nonzeroy)
            good_right_x, good_right_y = self.pixels_in_window(center_right, margin, window_height, nonzerox, nonzeroy)
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > minpix:
                leftx_current = np.int32(np.mean(good_left_x))

            if len(good_right_x) > minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img, leftx, lefty, rightx, righty):

        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3

        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))

        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))

        ploty = np.linspace(miny, maxy, img.shape[0])

        if len(lefty) > 500: # if len(lefty) > 0 and len(leftx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = [0, 0, 0]  

        if len(righty) > 500: # if len(righty) > 0 and len(rightx) > 0:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = [0, 0, 0]  

        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        out_img = np.dstack((img, img, img))

        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

        return out_img

    def measure_curvature(self, left_fit, right_fit):

        ym = 30 / 720
        xm = 3.7 / 700

        y_eval = 700 * ym

        left_curveR = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.absolute(2 * left_fit[0])
        right_curveR = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.absolute(2 * right_fit[0])

        xl = np.dot(left_fit, [700**2, 700, 1])
        xr = np.dot(right_fit, [700**2, 700, 1])
        pos = (1280 // 2 - (xl + xr) // 2) * xm

        return left_curveR, right_curveR, pos


