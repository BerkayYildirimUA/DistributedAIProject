
import cv2
import numpy as np


class BirdVisualiser:
    def __init__(self,width,height):
        self.width = width
        self.height = height

        # Define a perspective transform matrix (manually chosen)
        # src: trapezoid in front-view; dst: rectangle for top view
        src = np.float32([
            [450, 400],  # top-left
            [850, 400],  # top-right
            [1000, 700],  # bottom-right
            [300, 700]  # bottom-left
        ])

        dst = np.float32([
            [400, 0],
            [800, 0],
            [800, 600],
            [400, 600]
        ])
        # Compute transform
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.class_colors=[
            (0, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 0, 0),
            (0, 128, 0),
            (0, 0, 128)
        ]
        self.window_name="Bird EYE"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.bird_img_default = np.ones((self.height, self.width, 3), dtype=np.uint8)*255

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
                object_middle_points.append(([(box[2]+box[0])/2,(box[3]+box[1])/2]))
                colors.append(self.class_colors[class_ids[i]])
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
            cv2.circle(bird_img,(x,y), 5,color, 2)
        cv2.imshow(self.window_name, bird_img)
        cv2.waitKey(1)
    def cleanup(self):
        cv2.destroyAllWindows()
