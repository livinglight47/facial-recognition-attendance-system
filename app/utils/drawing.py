import cv2

def draw_detections(frame, detections, color=(0, 255, 0), show_label=True, show_conf=False):
    """
    detections: iterable of dicts with keys x1,y1,x2,y2,conf,cls_name
    """
    for d in detections:
        x1, y1, x2, y2 = map(int, (d["x1"], d["y1"], d["x2"], d["y2"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if show_label:
            label = d["cls_name"]
            if show_conf:
                label = f"{label} {d['conf']:.2f}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame
