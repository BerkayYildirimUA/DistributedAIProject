
import cv2
import numpy as np


class BirdVisualiser:
    def __init__(self,width,height):
        self.w = width
        self.h = height
        self.top_y = 0.65
        self.bottom_y = 1.00
        self.left_top_x = 0.25
        self.right_top_x = 0.75
        self.left_bottom_x = 0.00
        self.right_bottom_x = 1.0
        # Define a perspective transform matrix (manually chosen)
        # src: trapezoid in front-view; dst: rectangle for top view
        src = np.float32([
            (self.w * self.left_top_x, self.h * self.top_y),
            (self.w * self.right_top_x, self.h * self.top_y),
            (self.w * self.right_bottom_x, self.h * self.bottom_y),
            (self.w * self.left_bottom_x, self.h * self.bottom_y)
        ])
        dst = np.float32([
            (self.w * 0.25, 0),
            (self.w * 0.75, 0),
            (self.w * 0.75, self.h),
            (self.w * 0.25, self.h)
        ])
        # Compute transform
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.class_colors=[
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128)
        ]
        self.window_name="Bird EYE"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.bird_img_default = np.ones((self.h, self.w, 3), dtype=np.uint8)*255

    # Helper function to warp points
    def warp_points(self,pts):
        if len(pts) == 0:
            return []
        pts = np.array(pts, dtype='float32').reshape(-1, 1,2)
        warped = cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)
        return warped

    def get_object_coords_and_colors(self,boxes,class_ids):
        object_middle_points=[]
        colors=[]
        for i,box in enumerate(boxes):
            if class_ids[i].item() in [0,1,2,3,5]:
                print("IN")
                object_middle_points.append([int((box[2]+box[0])/2),int((box[3]+box[1])/2)])
                colors.append(self.class_colors[int(class_ids[i])])
        print(object_middle_points,colors)
        return object_middle_points,colors

    def generate_new_coords_with_colors(self,boxes,class_ids,lanes):
        # Warp everything
        lanes = self.warp_points(lanes)
        objects,colors =self.get_object_coords_and_colors(boxes,class_ids)
        objects = self.warp_points(objects)
        lane_colors=[(0,0,0)]*len(lanes)
        return [*lanes,*objects],[*colors,*lane_colors]

    def show(self,boxes,class_ids,lanes):
        coords, colors = self.generate_new_coords_with_colors(boxes,class_ids,lanes)
        bird_img=self.bird_img_default.copy()
        for co, color in zip(coords, colors):
            x,y=int(co[0]),int(co[1])
            if color != (255,255,255):
                cv2.circle(bird_img,(x,y), 20,color, -1)
            cv2.circle(bird_img,(x,y), 5,color, -1)

        cv2.imshow(self.window_name, bird_img)
        cv2.waitKey(1)
    def cleanup(self):
        cv2.destroyAllWindows()
