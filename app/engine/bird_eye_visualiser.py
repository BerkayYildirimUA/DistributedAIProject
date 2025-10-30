# from time import sleep
#
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# from matplotlib.animation import FuncAnimation
#
#
# class BirdVisualiser:
#     def __init__(self):
#
#         # Define a perspective transform matrix (manually chosen)
#         # src: trapezoid in front-view; dst: rectangle for top view
#         src = np.float32([
#             [450, 400],  # top-left
#             [850, 400],  # top-right
#             [1000, 700],  # bottom-right
#             [300, 700]  # bottom-left
#         ])
#
#         dst = np.float32([
#             [400, 0],
#             [800, 0],
#             [800, 600],
#             [400, 600]
#         ])
#         # Compute transform
#         self.M = cv2.getPerspectiveTransform(src, dst)
#         self.class_colors=["r","g","b","c","y","m"]
#
#         self.fig, self.ax = plt.subplots(figsize=(6, 8))
#         self.ax.invert_yaxis()
#         self.ax.axis('equal')
#         self.ax.set_title("Simplified Bird’s-Eye View")
#         plt.ion()  # interactive mode
#
#     # Helper function to warp points
#     def warp_points(self,pts):
#         if len(pts) == 0:
#             return []
#         print(pts)
#         pts = np.array(pts, dtype='float32').reshape(-1, 1,2)
#         print(pts.shape)
#         print(pts)
#         warped = cv2.perspectiveTransform(pts, self.M)[0].reshape(-1, 2)
#         print("wrapped",warped)
#         return warped
#
#     def get_object_coords_and_colors(self,boxes,class_ids):
#         object_middle_points=[]
#         colors=[]
#         for i,box in enumerate(boxes):
#             if class_ids[i] in [0,1,2,3,5]:
#                 object_middle_points.append(([(box[2]+box[0])/2,(box[3]+box[1])/2]))
#                 colors.append(self.class_colors[class_ids[i]])
#         return object_middle_points,colors
#     def generate_new_coords_with_colors(self,boxes,class_ids,lane_l,lane_r):
#         # Warp everything
#         lane_l = self.warp_points(lane_l)
#         lane_r = self.warp_points(lane_r)
#         print(lane_l, lane_r)
#         objects,colors =self.get_object_coords_and_colors(boxes,class_ids)
#         objects = self.warp_points(objects)
#         lane_colors=["k"]*len(lane_l)+["k"]*len(lane_r)
#         return [*lane_l,*lane_r,*objects],[*colors,*lane_colors]
#
#     def show(self,boxes,class_ids,lane_l,lane_r):
#         coords, colors = self.generate_new_coords_with_colors(boxes,class_ids,lane_l,lane_r)
#
#         self.ax.clear()
#         self.ax.cla()  # clear previous points
#         self.ax.invert_yaxis()
#         self.ax.axis('equal')
#         self.ax.set_title("Simplified Bird’s-Eye View")
#         self.fig.canvas.flush_events()  # make sure update is visible
#
#         # Draw objects
#         for co,color in zip(coords,colors):
#             x,y=co[0],co[1]
#             self.ax.scatter(x, y, color=color)
#         self.fig.canvas.draw()
#         # plt.pause(0.05)
#
#     def cleanup(self):
#         plt.close('all')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import numpy as np

class BirdVisualiser:
    def __init__(self):
        # Perspective transform (front view → bird’s-eye view)
        src = np.float32([[450, 400], [850, 400], [1000, 700], [300, 700]])
        dst = np.float32([[400, 0], [800, 0], [800, 600], [400, 600]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.class_colors = ["r", "g", "b", "c", "y", "m"]

        # Create figure & axes once
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        self.ax.invert_yaxis()
        self.ax.axis('equal')
        self.ax.set_title("Simplified Bird’s-Eye View")

        # Prepare scatter object (initially empty)
        self.scatter = self.ax.scatter([], [])

    # Warp points to bird’s-eye view
    def warp_points(self, pts):
        if len(pts) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)
        return warped

    # Compute object centers
    def get_object_coords_and_colors(self, boxes, class_ids):
        centers, colors = [], []
        for i, box in enumerate(boxes):
            if class_ids[i] in [0, 1, 2, 3, 5]:
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                centers.append([x_center, y_center])
                colors.append(self.class_colors[class_ids[i]])
        return centers, colors

    # Warp lanes and objects, get colors
    def generate_new_coords_with_colors(self, boxes, class_ids, lane_l, lane_r):
        lane_l_warp = self.warp_points(lane_l)
        lane_r_warp = self.warp_points(lane_r)
        objects, obj_colors = self.get_object_coords_and_colors(boxes, class_ids)
        objects_warp = self.warp_points(objects)
        lane_colors = ["k"] * (len(lane_l_warp) + len(lane_r_warp))
        all_coords = np.vstack([lane_l_warp, lane_r_warp, objects_warp])
        all_colors = lane_colors + obj_colors
        return all_coords, all_colors

    # Update function for FuncAnimation
    def update(self, frame_data):
        """
        frame_data = (boxes, class_ids, lane_l, lane_r)
        """
        boxes, class_ids, lane_l, lane_r = frame_data
        coords, colors = self.generate_new_coords_with_colors(boxes, class_ids, lane_l, lane_r)
        if len(coords) > 0:
            self.scatter.set_offsets(coords)
            self.scatter.set_color(colors)
        return self.scatter,

    # Run animation
    def animate(self, frames_data, interval=50):
        """
        frames_data: list of tuples (boxes, class_ids, lane_l, lane_r)
        interval: ms between frames
        """
        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=frames_data,
            interval=interval,
            blit=True,
            repeat=False
        )
        plt.show()


