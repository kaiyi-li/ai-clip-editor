#!/usr/bin/env python3
import argparse
import numpy as np
import faiss

def build_index(features_npz, out_index_path):
    obj = np.load(features_npz, allow_pickle=True)
    feats = obj['features']  # [N, D]
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    feats_norm = feats / norms

    d = feats_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(feats_norm.astype('float32'))
    faiss.write_index(index, out_index_path)
    np.savez_compressed(features_npz.replace('.npz','') + '_norm.npz', **obj, features=feats_norm)
    print(f"faiss index 写入到 {out_index_path}, 含 {index.ntotal} 向量")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--index", required=True)
    args = p.parse_args()
    build_index(args.features, args.index)
