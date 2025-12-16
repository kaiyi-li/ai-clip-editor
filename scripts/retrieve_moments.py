#!/usr/bin/env python3
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
from scipy.ndimage import gaussian_filter1d
import argparse

def load_data(features_norm_npz, index_path):
    obj = np.load(features_norm_npz, allow_pickle=True)
    starts = obj['windows_start'].tolist()
    ends = obj['windows_end'].tolist()
    feats = obj['features'].astype('float32')
    meta = dict(obj['meta'].tolist())
    index = faiss.read_index(index_path)
    return starts, ends, feats, meta, index

def encode_text(text, model, processor, device='cpu'):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        txt_feat = model.get_text_features(**inputs)
    v = txt_feat.cpu().numpy()[0]
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype('float32')

def merge_scores_to_intervals(starts, ends, scores, smooth_sigma=1.0, score_thresh=0.2, min_len_sec=0.5):
    s = gaussian_filter1d(scores, sigma=smooth_sigma)
    mask = s >= score_thresh
    intervals = []
    N = len(s)
    i = 0
    while i < N:
        if mask[i]:
            j = i
            while j+1 < N and mask[j+1]:
                j += 1
            start = starts[i]
            end = ends[j]
            avg_score = float(s[i:j+1].mean())
            if end - start >= min_len_sec:
                intervals.append((start, end, avg_score))
            i = j+1
        else:
            i += 1
    intervals.sort(key=lambda x: x[2], reverse=True)
    return intervals

def retrieve(query, features_norm_npz, index_path, topk=10, device='cpu'):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    starts, ends, feats, meta, index = load_data(features_norm_npz, index_path)
    text_v = encode_text(query, model, processor, device)
    scores = (feats @ text_v).astype(float)
    intervals = merge_scores_to_intervals(starts, ends, scores, smooth_sigma=1.0, score_thresh=0.25, min_len_sec=0.5)
    return intervals, scores

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--features", required=True, help="norm npz, e.g. demo_features_norm.npz")
    p.add_argument("--query", required=True)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    intervals, scores = retrieve(args.query, args.features, args.index, args.topk, args.device)
    print("检索到的时间段（按 score 降序）：")
    for st, ed, sc in intervals[:10]:
        print(f"{st:.2f}s -> {ed:.2f}s  score={sc:.4f}")
