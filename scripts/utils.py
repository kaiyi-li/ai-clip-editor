import os
import cv2
import numpy as np

def get_video_fps(path):
    vcap = cv2.VideoCapture(path)
    fps = vcap.get(cv2.CAP_PROP_FPS)
    vcap.release()
    return fps if fps>0 else 25.0

def frames_from_video(video_path, sample_fps=1.0):
    """
    按 sample_fps 采样，返回 list of (time_sec, frame_image BGR).
    """
    vcap = cv2.VideoCapture(video_path)
    fps = vcap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    frames = []
    idx = 0
    while True:
        ret, frame = vcap.read()
        if not ret:
            break
        if idx % step == 0:
            time_sec = idx / fps
            frames.append((time_sec, frame))
        idx += 1
    vcap.release()
    return frames
