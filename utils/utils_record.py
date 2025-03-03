from multiprocessing import Queue

import cv2
import numpy as np


def export_to_cam(images: Queue, fps=60, reso=[1920, 1080], export_path="output/robot_cam.mp4"):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(export_path, fourcc, 60, reso)
    length_each_frame = int(60 / fps)
    while (frame := images.get()) is not None:
        frame = (frame).astype(np.uint8)
        for _ in range(length_each_frame):
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Release the video writer
    video_writer.release()
    print(f"Video saved as {export_path}")
