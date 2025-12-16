#!/usr/bin/env python3
import argparse
import numpy as np
from scripts.utils import frames_from_video, get_video_fps
import torch
from transformers import CLIPProcessor, CLIPModel
import cv2

def extract_window_features(video_path, out_npz, window_sec=2.0, stride_sec=1.0, sample_fps=1.0, device='cpu'):
    fps = get_video_fps(video_path)
    frames = frames_from_video(video_path, sample_fps=sample_fps)
    if len(frames) == 0:
        raise RuntimeError("没有抽到任何帧，请检查视频路径或 sample_fps 参数。")
    times = [t for t,_ in frames]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for _,img in frames]

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    window_frames = []
    starts = []
    ends = []
    t = 0.0
    max_t = times[-1] if times else 0.0
    while t <= max_t:
        start = t
        end = t + window_sec
        idxs = [i for i,tt in enumerate(times) if tt>=start and tt<end]
        if len(idxs) > 0:
            batch_imgs = [imgs[i] for i in idxs]
            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                image_embeds = model.get_image_features(**inputs)  # [B, D]
                image_embeds = image_embeds.cpu().numpy()
            avg = image_embeds.mean(axis=0)
            window_frames.append(avg)
            starts.append(start)
            ends.append(min(end, max_t))
        t += stride_sec

    if len(window_frames)==0:
        raise RuntimeError("没有抽到任何窗口，请检查参数或视频长度。")

    feats = np.stack(window_frames, axis=0)  # [N, D]
    np.savez_compressed(out_npz, windows_start=np.array(starts), windows_end=np.array(ends), features=feats,
                        meta={'fps': fps, 'window_sec': window_sec, 'stride_sec': stride_sec, 'sample_fps': sample_fps})
    print(f"保存特征到 {out_npz}, 窗口数={feats.shape[0]}, 维度={feats.shape[1]}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--window_sec", type=float, default=2.0)
    p.add_argument("--stride_sec", type=float, default=1.0)
    p.add_argument("--fps", type=float, dest="sample_fps", default=1.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    extract_window_features(args.video, args.out, args.window_sec, args.stride_sec, args.sample_fps, args.device)
