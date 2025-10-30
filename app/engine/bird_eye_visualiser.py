from time import sleep

import matplotlib.pyplot as plt
import cv2
import numpy as np
# plt.ion()  # turn on interactive mode


class BirdVisualiser:
    def __init__(self):

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
        self.class_colors=["r","g","b","c","y","m"]
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.ax.invert_yaxis()
        self.ax.axis('equal')
        self.ax.set_title("Simplified Bird’s-Eye View")
    # Helper function to warp points
    def warp_points(self,pts):
        if len(pts) == 0:
            return []
        print(pts)
        pts = np.array(pts, dtype='float32').reshape(-1, 1,2)
        print(pts.shape)
        print(pts)
        warped = cv2.perspectiveTransform(pts, self.M)[0].reshape(-1, 2)
        print("wrapped",warped)
        return warped

    def get_object_coords_and_colors(self,boxes,class_ids):
        object_middle_points=[]
        colors=[]
        for i,box in enumerate(boxes):
            if class_ids[i] in [0,1,2,3,5]:
                object_middle_points.append(([(box[2]-box[0])/2,(box[3]-box[1])/2]))
                colors.append(self.class_colors[class_ids[i]])
        return object_middle_points,colors
    def generate_new_coords_with_colors(self,boxes,class_ids,lane_l,lane_r):
        # Warp everything
        lane_l = self.warp_points(lane_l)
        lane_r = self.warp_points(lane_r)
        print(lane_l, lane_r)
        objects,colors =self.get_object_coords_and_colors(boxes,class_ids)
        objects = self.warp_points(objects)
        lane_colors=["k"]*len(lane_l)+["k"]*len(lane_r)
        return [*lane_l,*lane_r,*objects],[*colors,*lane_colors]

    def show(self,boxes,class_ids,lane_l,lane_r):
        coords, colors = self.generate_new_coords_with_colors(boxes,class_ids,lane_l,lane_r)

        # Draw objects
        for co,color in zip(coords,colors):
            x,y=co[0],co[1]
            self.ax.scatter(x, y, color=color)
        self.fig.canvas.draw()
        plt.pause(0.05)

    def cleanup(self):
        plt.close('all')
