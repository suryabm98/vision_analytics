import cv2
import numpy as np

def letterbox_resize(frame, target_w=1280, target_h=960):
    """Preserves aspect ratio and prevents facial geometry distortion."""
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=frame.dtype)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


class LightweightFaceTracker:
    """IoU-based tracker matching faces across consecutive frames."""
    def __init__(self, max_disappeared=5, iou_threshold=0.3):
        self.next_object_id = 1
        self.objects = {}         # {track_id: bbox}
        self.disappeared = {}     # {track_id: frames_missed}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    @staticmethod
    def _compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detected_boxes):
        if len(detected_boxes) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.objects[track_id]
                    del self.disappeared[track_id]
            return {}

        if len(self.objects) == 0:
            result = {}
            for idx, box in enumerate(detected_boxes):
                tid = f"ID-{self.next_object_id}"
                self.objects[tid] = box
                self.disappeared[tid] = 0
                result[idx] = tid
                self.next_object_id += 1
            return result

        object_ids = list(self.objects.keys())
        previous_boxes = list(self.objects.values())

        iou_matrix = np.zeros((len(previous_boxes), len(detected_boxes)), dtype=np.float32)
        for i, p_box in enumerate(previous_boxes):
            for j, d_box in enumerate(detected_boxes):
                iou_matrix[i, j] = self._compute_iou(p_box, d_box)

        matched_indices = []
        if iou_matrix.size > 0:
            for _ in range(min(len(previous_boxes), len(detected_boxes))):
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                if iou_matrix[i, j] < self.iou_threshold:
                    break
                matched_indices.append((i, j))
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        matched_p = {i for i, _ in matched_indices}
        matched_d = {j for _, j in matched_indices}

        result = {}
        for i, j in matched_indices:
            tid = object_ids[i]
            self.objects[tid] = detected_boxes[j]
            self.disappeared[tid] = 0
            result[j] = tid

        for i, tid in enumerate(object_ids):
            if i not in matched_p:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.objects[tid]
                    del self.disappeared[tid]

        for j, box in enumerate(detected_boxes):
            if j not in matched_d:
                tid = f"ID-{self.next_object_id}"
                self.objects[tid] = box
                self.disappeared[tid] = 0
                result[j] = tid
                self.next_object_id += 1

        return result