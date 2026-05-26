"""
CRAFT 검출 후처리 — 추론 전용 함수만 발췌.

원본: AI Hub 공공 OCR 데이터 — craft/misc.py (MIT License)
data_manipulation.py 의 CUDA OpenCV 의존을 제거하고,
generate_word_bbox_batch / merge_adjacent_boxes 등 추론에 필요한 함수만 포함.
"""
from __future__ import annotations

import math
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def order_points(box: np.ndarray) -> np.ndarray:
    x_sorted_arg = np.argsort(box[:, 0])
    if box[x_sorted_arg[0], 1] > box[x_sorted_arg[1], 1]:
        tl = x_sorted_arg[1]
    else:
        tl = x_sorted_arg[0]
    return np.array([box[(tl + i) % 4] for i in range(4)])


def _link_to_word_bbox(to_find, word_bbox):
    if len(word_bbox) == 0:
        return [np.zeros([0, 4, 1, 2], dtype=np.int32)]

    word_sorted_character: list[list] = [[] for _ in word_bbox]
    compare_word_polygons = []
    for word in word_bbox:
        compare_word_polygons.append(
            Polygon(word.reshape([word.shape[0], 2])).buffer(0)
        )
    tree = STRtree(compare_word_polygons)

    for cont in to_find:
        if cont.shape[0] < 4:
            continue
        rectangle = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rectangle)
        if Polygon(box).area == 0:
            continue
        ordered_bbox = order_points(box)
        a = Polygon(cont.reshape([cont.shape[0], 2])).buffer(0)
        if a.area == 0:
            continue

        ratio_arr = []
        for b in tree.query_items(a):
            ratio_arr.append(
                (tree._geoms[b].intersection(a).area / a.area, b)
            )
        ratio_arr = sorted(ratio_arr, key=lambda x: (-x[0], x[1]))
        if ratio_arr:
            idx = 0 if ratio_arr[0][0] == 0.0 else ratio_arr[0][1]
            word_sorted_character[idx].append(ordered_bbox)
        else:
            word_sorted_character[0].append(ordered_bbox)

    return [
        np.array(w, dtype=np.int32).reshape([len(w), 4, 1, 2])
        for w in word_sorted_character
    ]


def generate_word_bbox(
    character_heatmap: np.ndarray,
    affinity_heatmap: np.ndarray,
    character_threshold: float,
    affinity_threshold: float,
    word_threshold: float,
) -> dict:
    img_h, img_w = character_heatmap.shape

    _, text_score = cv2.threshold(character_heatmap, character_threshold, 1, 0)
    _, link_score = cv2.threshold(affinity_heatmap, affinity_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    det = []
    mapper = []
    for k in range(1, n_labels):
        try:
            size = stats[k, cv2.CC_STAT_AREA]
            if size < 10:
                continue
            where = labels == k
            if np.max(character_heatmap[where]) < word_threshold:
                continue

            seg_map = np.zeros(character_heatmap.shape, dtype=np.uint8)
            seg_map[where] = 255
            seg_map[np.logical_and(link_score == 1, text_score == 0)] = 0

            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
            sx = max(x - niter, 0)
            sy = max(y - niter, 0)
            ex = min(x + w + niter + 1, img_w)
            ey = min(y + h + niter + 1, img_h)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
            seg_map[sy:ey, sx:ex] = cv2.dilate(seg_map[sy:ey, sx:ex], kernel)

            np_contours = (
                np.roll(np.array(np.where(seg_map != 0)), 1, axis=0)
                .transpose()
                .reshape(-1, 2)
            )
            rectangle = cv2.minAreaRect(np_contours)
            box = cv2.boxPoints(rectangle)

            bw, bh = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(bw, bh) / (min(bw, bh) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
                t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

            start_idx = box.sum(axis=1).argmin()
            box = np.roll(box, 4 - start_idx, 0)
            det.append(np.array(box))
            mapper.append(k)
        except Exception:
            continue

    char_contours, _ = cv2.findContours(
        text_score.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    affinity_contours, _ = cv2.findContours(
        link_score.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    char_contours = _link_to_word_bbox(char_contours, det)
    affinity_contours = _link_to_word_bbox(affinity_contours, det)

    return {
        "word_bbox": np.array(det, dtype=np.int32).reshape([len(det), 4, 1, 2]),
        "characters": char_contours,
        "affinity": affinity_contours,
    }


def generate_word_bbox_batch(
    batch_character_heatmap: np.ndarray,
    batch_affinity_heatmap: np.ndarray,
    character_threshold: float,
    affinity_threshold: float,
    word_threshold: float,
) -> list:
    result = []
    for i in range(batch_character_heatmap.shape[0]):
        returned = generate_word_bbox(
            batch_character_heatmap[i],
            batch_affinity_heatmap[i],
            character_threshold,
            affinity_threshold,
            word_threshold,
        )
        result.append(returned["word_bbox"])
    return result


def _is_in_box(point: tuple, box: np.ndarray) -> bool:
    x, y = point
    return bool(box[0, 0] <= x <= box[1, 0] and box[0, 1] <= y <= box[1, 1])


def _merge_two_box(rbox: np.ndarray, lbox: np.ndarray) -> np.ndarray:
    new_box = np.zeros_like(rbox)
    new_box[0, 0] = min(rbox[0, 0], lbox[0, 0])
    new_box[0, 1] = min(rbox[0, 1], lbox[0, 1])
    new_box[1, 0] = max(rbox[1, 0], lbox[1, 0])
    new_box[1, 1] = max(rbox[1, 1], lbox[1, 1])
    return new_box.reshape(1, 2, 2)


def merge_adjacent_boxes(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
    affinity_map: np.ndarray,
    direction: str,
    threshold: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """사분면 경계에서 인접 박스를 병합."""
    assert direction in ("left", "up")
    half = affinity_map.shape[1] // 2
    to_be_merged = []

    if direction == "left":
        for i in range(len(boxes1)):
            lx = boxes1[i, 0, 0]
            if lx < half + 0.01 * half:
                my = int(boxes1[i, :, 1].mean())
                if affinity_map[my, lx] > threshold:
                    point = (half - 1, my)
                    for ii in range(len(boxes2)):
                        if _is_in_box(point, boxes2[ii]):
                            to_be_merged.append((i, ii))
                            break
    else:
        for i in range(len(boxes1)):
            ly = boxes1[i, 0, 1]
            if ly < half + 0.01 * half:
                mx = int(boxes1[i, :, 0].mean())
                if affinity_map[mx, ly] > threshold / 20:
                    point = (mx, half - 1)
                    for ii in range(len(boxes2)):
                        if _is_in_box(point, boxes2[ii]):
                            to_be_merged.append((i, ii))
                            break

    merged_boxes = np.empty((0, 2, 2))
    for i, ii in to_be_merged:
        new_box = _merge_two_box(boxes1[i], boxes2[ii])
        merged_boxes = np.concatenate((merged_boxes, new_box))

    boxes1 = np.delete(boxes1, [x[0] for x in to_be_merged], axis=0)
    boxes2 = np.delete(boxes2, [x[1] for x in to_be_merged], axis=0)

    return boxes1, boxes2, merged_boxes
