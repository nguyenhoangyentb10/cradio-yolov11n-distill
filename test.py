import os
import cv2
from ultralytics import YOLO

weights = "/mnt/f/Intership/Smoke/smoking_pipeline/models/det_best.pt"
img_path = "/mnt/f/Intership/Smoke/smoking_pipeline/input/img2.png"
out_path = "/mnt/f/Intership/Smoke/smoking_pipeline/output/out_det_best.jpg"

conf = 0.25
iou = 0.45
imgsz = 640

def run_det(weights, img_path, out_path, conf=0.25, iou=0.45, imgsz=640):
    assert os.path.exists(weights), f"Không thấy weights: {weights}"
    assert os.path.exists(img_path), f"Không thấy ảnh: {img_path}"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    model = YOLO(weights)
    results = model.predict(source=img_path, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    r = results[0]

    plotted = r.plot()  # ảnh đã vẽ bbox (BGR)
    cv2.imwrite(out_path, plotted)
    print(f"[OK] Saved annotated image -> {out_path}")

    # in thông tin bbox
    if r.boxes is None or len(r.boxes) == 0:
        print("[INFO] Không detect được bbox nào (thử giảm conf xuống 0.1).")
        return

    names = r.names
    print(f"[INFO] Found {len(r.boxes)} boxes:")
    for i, b in enumerate(r.boxes):
        cls_id = int(b.cls.item())
        cls_name = names[cls_id] if isinstance(names, list) else names.get(cls_id, str(cls_id))
        score = float(b.conf.item())
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        print(f"  #{i}: {cls_name} conf={score:.3f} xyxy=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")

if __name__ == "__main__":
    run_det(weights, img_path, out_path, conf=conf, iou=iou, imgsz=imgsz)

