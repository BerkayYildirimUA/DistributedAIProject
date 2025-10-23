import carla
import numpy as np
import cv2
import time
import threading

class LaneDetection:
    def __init__(self):
        # Initial perspective parameters. These were obtained experimentally, but may need to be changed if
        # performance is poor
        # top_y=0.65, bottom_y=1.00, ltx=0.25, rtx=0.75, lbx=0.00, rbx=1.00
        self.top_y = 0.65
        self.bottom_y = 1.00
        self.left_top_x = 0.25
        self.right_top_x = 0.75
        self.left_bottom_x = 0.00
        self.right_bottom_x = 1.0
        self.prev_hist = None
        self.hist_smooth_factor = 0.8
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.fit_smooth_factor = 0.8  # higher = smoother lane, slower reaction



    def draw_lane(self, binary_warped, Minv, original_frame):
        # Find nonzero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Fit a 2nd order polynomial to left and right lane pixels
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # Temporal histogram smoothing... does not really seem to work properly
        if self.prev_hist is not None:
            histogram = (self.hist_smooth_factor * self.prev_hist + (1 - self.hist_smooth_factor) * histogram)

        self.prev_hist = histogram.copy()
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 3 # Controls vertical window size
        margin = 50 # Margin of error for the sliding window
        minpix = 50 # Detection sensitivity
        window_height = binary_warped.shape[0]//nwindows

        leftx_current = leftx_base
        rightx_current = rightx_base
        leftx, lefty, rightx, righty = [], [], [], []

        # Sliding window algorithm
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            leftx.extend(nonzerox[good_left_inds])
            lefty.extend(nonzeroy[good_left_inds])
            rightx.extend(nonzerox[good_right_inds])
            righty.extend(nonzeroy[good_right_inds])

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Not enough points for a reliable fit
        if len(leftx) < 100 or len(rightx) < 100:
            return original_frame
        
        # Fit polynomials
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)   

        # Store current fits for next frame. Attempt at creating temporal smoothing
        self.prev_left_fit = left_fit
        self.prev_right_fit = right_fit

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw the lane
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

        # Warp back to original image
        newwarp = cv2.warpPerspective(color_warp, Minv, (original_frame.shape[1], original_frame.shape[0]))
        result = cv2.addWeighted(original_frame, 1, newwarp, 0.3, 0)

        return result

    def get_src_dst(self, w, h):
        src = np.float32([
            (w * self.left_top_x, h * self.top_y),
            (w * self.right_top_x, h * self.top_y),
            (w * self.right_bottom_x, h * self.bottom_y),
            (w * self.left_bottom_x, h * self.bottom_y)
        ])
        dst = np.float32([
            (w * 0.25, 0),
            (w * 0.75, 0),
            (w * 0.75, h),
            (w * 0.25, h)
        ])
        return src, dst

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
        h, w = img.shape[:2]
        src, dst = self.get_src_dst(w, h)
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv, src, dst

    def warp_perspective(self, img, M):
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    def binary_threshold(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        white_mask = cv2.inRange(hsv, (0, 0, 200), (255, 30, 255))
        color_mask = cv2.bitwise_or(yellow_mask, white_mask)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        combined = cv2.bitwise_or(color_mask, edges)
        kernel_fill = np.ones((5,5), np.uint8)  # fill small gaps
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_fill, iterations=1)

        kernel_erode = np.ones((3,3), np.uint8)  # erode edges
        combined = cv2.erode(combined, kernel_erode, iterations=1)

        # Zero out the sides to focus on central lanes 
        h, w = combined.shape[:2]
        side_crop = int(0.15 * w)  # adjust fraction to control how much of the sides to ignore
        mask = np.zeros_like(combined, dtype=np.uint8)
        mask[:, side_crop:w - side_crop] = 255  # keep only the central region
        combined = cv2.bitwise_and(combined, mask)

        # Zero out the bottom part to remove hood/edge noise
        bottom_crop = int(0.12 * h)  # tweak this % if needed
        combined[h - bottom_crop:, :] = 0
        return combined


    def process_frame(self, frame_bgr):
        M, Minv, src, dst = self.get_perspective_matrices(frame_bgr)
        warped = self.warp_perspective(frame_bgr, M)
        binary = self.binary_threshold(warped)
        return warped, binary, src, Minv

# -----------------------------
# Carla Environment Setup
# -----------------------------
def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world("Town04")
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    bp_vehicle = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[10]
    ego_vehicle = world.try_spawn_actor(bp_vehicle, spawn_point)
    if not ego_vehicle:
        print("Could not spawn vehicle!")
        return

    traffic_manager = client.get_trafficmanager()
    for vehicle in world.get_actors().filter('*vehicle*'):
        if vehicle.id != ego_vehicle.id:
            vehicle.set_autopilot(True,traffic_manager.get_port())
    
    ego_vehicle.set_autopilot(True, traffic_manager.get_port())

    camera_init_trans = carla.Transform(carla.Location(x=1.7, z=1.5))
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "640")
    camera_bp.set_attribute("image_size_y", "480")
    camera_bp.set_attribute("sensor_tick", "0.05")

    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
    lane_detector = LaneDetection()
    stop_event = threading.Event()

    def process_image(image):
        if stop_event.is_set():
            return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        frame = array.reshape((image.height, image.width, 4))[:, :, :3]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        warped, binary, src, Minv = lane_detector.process_frame(frame_bgr)
        overlayed_frame = lane_detector.draw_lane(binary, Minv, frame_bgr)

        # Draw trapezoid on original
        overlay = frame_bgr.copy()
        cv2.polylines(overlay, [np.int32(src)], isClosed=True, color=(0, 0, 255), thickness=2)

        cv2.imshow("Lane Overlay", overlayed_frame)
        #cv2.imshow("Warped Bird's Eye", warped)
        cv2.imshow("Binary Threshold", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            stop_event.set()

    camera.listen(lambda image: process_image(image))

    try:
        while not stop_event.is_set():
            time.sleep(0.05)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        print("Cleaning up...")
        camera.stop()
        camera.destroy()
        ego_vehicle.destroy()
        cv2.destroyAllWindows()
        print("Shutdown complete.")

if __name__ == '__main__':
    main()
