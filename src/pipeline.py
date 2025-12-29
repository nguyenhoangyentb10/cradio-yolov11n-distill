import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def xyxy_to_int(xyxy, w, h):
    x1, y1, x2, y2 = map(float, xyxy)
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def crop_image(img_bgr, box_xyxy):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy_to_int(box_xyxy, w, h)
    return img_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def pick_boxes(det_res, target_class_names=None):
    """
    Return list of dict boxes filtered by class names
    det_res: Results for single image (detection)
    target_class_names: set[str] or None (None -> take all)
    """
    boxes = det_res.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = det_res.names
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    out = []
    for i in range(len(cls)):
        cls_id = int(cls[i])
        cls_name = names.get(cls_id, str(cls_id))
        if (target_class_names is None) or (cls_name in target_class_names):
            out.append({
                "cls_id": cls_id,
                "cls_name": cls_name,
                "conf": float(conf[i]),
                "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
            })
    return out


def run(
    det_model_path,
    cls_model_path,
    image_path,
    out_dir,
    smoking_action_class="smoking",
    cigarette_class="cigarette",
    cls_positive_labels=("smoking", "smoke", "cigarette", "hút_thuốc"),
    cls_threshold=0.6,
    device=None,  # "cpu", "0", etc.
):
    out_dir = Path(out_dir)
    json_dir = out_dir / "json"
    crops_dir = out_dir / "crops"
    ann_dir = out_dir / "annotated"
    ensure_dir(json_dir)
    ensure_dir(crops_dir)
    ensure_dir(ann_dir)

    image_path = Path(image_path)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img_bgr.shape[:2]

    det_model = YOLO(det_model_path)
    cls_model = YOLO(cls_model_path)

    # --- A) Detection ---
    det_results = det_model.predict(source=str(image_path), verbose=False, device=device)
    det_res = det_results[0]

    det_json_str = det_res.to_json()
    (json_dir / f"{image_path.stem}_det.json").write_text(det_json_str, encoding="utf-8")

    smoking_boxes = pick_boxes(det_res, {smoking_action_class})
    cig_boxes = pick_boxes(det_res, {cigarette_class})

    has_smoking_action = len(smoking_boxes) > 0

    # crop rule: prefer cigarette bbox; else smoking bbox
    crop_candidates = cig_boxes if len(cig_boxes) > 0 else smoking_boxes

    crop_meta = []
    crop_paths = []
    for idx, b in enumerate(crop_candidates):
        crop, (x1, y1, x2, y2) = crop_image(img_bgr, b["bbox_xyxy"])
        crop_path = crops_dir / f"{image_path.stem}_crop_{idx}.jpg"
        cv2.imwrite(str(crop_path), crop)
        crop_paths.append(str(crop_path))
        crop_meta.append({
            "crop_path": str(crop_path),
            "from_cls": b["cls_name"],
            "from_bbox_xyxy": [x1, y1, x2, y2],
            "from_conf": b["conf"],
        })

    # --- B) Classification ---
    cls_outputs = []
    confirmed = False

    positive_set = set(cls_positive_labels)

    for crop_path in crop_paths:
        cls_results = cls_model.predict(source=crop_path, verbose=False, device=device)
        cls_res = cls_results[0]

        probs = cls_res.probs
        top1_id = int(probs.top1)
        top1_conf = float(probs.top1conf)
        label = cls_res.names.get(top1_id, str(top1_id))

        is_smoking = (label in positive_set) and (top1_conf >= cls_threshold)
        if is_smoking:
            confirmed = True

        cls_outputs.append({
            "crop_path": crop_path,
            "top1_label": label,
            "top1_conf": top1_conf,
            "is_smoking": bool(is_smoking),
        })

    # annotated image (only if confirmed)
    annotated_path = None
    if confirmed:
        plotted = det_res.plot()  # BGR ndarray with boxes
        annotated_path = str(ann_dir / f"{image_path.stem}_confirmed.jpg")
        cv2.imwrite(annotated_path, plotted)

    final = {
        "image": str(image_path),
        "image_size": {"w": w, "h": h},
        "models": {"det": str(det_model_path), "cls": str(cls_model_path)},
        "detection": {
            "has_smoking_action": bool(has_smoking_action),
            "smoking_boxes": smoking_boxes,
            "cigarette_boxes": cig_boxes,
        },
        "crops": crop_meta,
        "classification": cls_outputs,
        "confirmed_smoking": bool(confirmed),
        "annotated_image": annotated_path,
    }

    (json_dir / f"{image_path.stem}_final.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return final


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="path to detection .pt")
    ap.add_argument("--cls", required=True, help="path to classification .pt")
    ap.add_argument("--img", required=True, help="path to input image")
    ap.add_argument("--out", default="/workspace/output", help="output dir")
    ap.add_argument("--device", default=None, help='e.g. "cpu" or "0" (GPU id)')





    ap.add_argument("--smoking_action_class", default="smoking")
    ap.add_argument("--cigarette_class", default="cigarette")
    ap.add_argument("--cls_positive_labels", default="smoking,smoke,cigarette,hút_thuốc")
    ap.add_argument("--cls_threshold", type=float, default=0.6)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    final = run(
        det_model_path=args.det,
        cls_model_path=args.cls,
        image_path=args.img,
        out_dir=args.out,
        smoking_action_class=args.smoking_action_class,
        cigarette_class=args.cigarette_class,
        cls_positive_labels=tuple(s.strip() for s in args.cls_positive_labels.split(",") if s.strip()),
        cls_threshold=args.cls_threshold,
        device=args.device,
    )
    print(json.dumps(final, ensure_ascii=False, indent=2))
