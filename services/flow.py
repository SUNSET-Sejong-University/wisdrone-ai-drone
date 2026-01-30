""" 
bbox 중심점 → 좌표
공간 clustering (K-means / grid)
각 cluster별: 사람 수, 평균 위치, 평균 이동 벡터
정체 cluster + 이동 cluster 관계 반환
"""
import math
import numpy as np
from typing import List, Dict
from sklearn.cluster import KMeans


def compute_cluster_flow(
    frame_results: List[Dict],
    n_clusters: int = 3
) -> Dict:
    """
    군집별 이동 + 정체 분석
    """

    if len(frame_results) < 2:
        return {"clusters": [], "pressure": 0.0}

    # ----------------------------
    # 1. 사람 프레임별 중심점 추출
    # ----------------------------
    all_centers = []
    frame_centers = []

    for frame in frame_results:
        centers = []
        for obj in frame.get("objects", []):
            x1, y1, x2, y2 = obj["bbox"]
            centers.append([(x1 + x2) / 2, (y1 + y2) / 2])
        frame_centers.append(centers)
        all_centers.extend(centers)

    if len(all_centers) < n_clusters:
        return {"clusters": [], "pressure": 0.0}

    X = np.array(all_centers)

    # ----------------------------
    # 2. 공간 군집화
    # ----------------------------
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(X)

    clusters = []
    idx = 0

    # ----------------------------
    # 3. 군집별 통계
    # ----------------------------
    for c in range(n_clusters):
        cluster_points = X[labels == c]
        count = len(cluster_points)

        if count == 0:
            clusters.append({
                "count": 0,
                "flow_mag": 0.0,
                "stagnant": True
            })
            continue

        # 프레임 t vs t+1 평균 이동
        deltas = []
        for t in range(len(frame_centers) - 1):
            if len(frame_centers[t]) == 0 or len(frame_centers[t+1]) == 0:
                continue

            prev = np.mean(frame_centers[t], axis=0)
            curr = np.mean(frame_centers[t+1], axis=0)
            deltas.append(curr - prev)

        if deltas:
            delta = np.mean(deltas, axis=0)
            mag = float(np.linalg.norm(delta))
        else:
            mag = 0.0

        clusters.append({
            "count": count,
            "flow_mag": round(mag, 2),
            "stagnant": mag < 5.0
        })

    # ----------------------------
    # 4. 압박 지표 (핵심)
    # ----------------------------
    stagnant = [c for c in clusters if c["stagnant"] and c["count"] > 10]
    moving = [c for c in clusters if not c["stagnant"] and c["count"] > 10]

    pressure = 0.0
    for s in stagnant:
        for m in moving:
            pressure += m["flow_mag"] * min(s["count"], m["count"])

    return {
        "clusters": clusters,
        "pressure": round(float(pressure), 2)
    }
