"""
Export MDv1000 redwood/sorrel models to ONNX and validate predictions.

Large exports may produce sidecar .onnx.data files when external tensor data is used.
"""

import argparse
import json
import os
import traceback
from dataclasses import asdict
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from megadetector.detection import pytorch_detector
from megadetector.detection.run_detector import try_download_known_detector
from megadetector.utils import ct_utils


@dataclass
class ValidationSummary:
    """Validation metrics for one model export."""

    model: str
    pt_model_path: str
    onnx_model_path: str
    pt_detection_count: int
    onnx_detection_count: int
    matched_pairs: int
    max_bbox_l1: float
    mean_bbox_l1: float
    max_conf_abs_diff: float
    mean_conf_abs_diff: float
    min_iou: float
    mean_iou: float
    is_close: bool


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


class _ExportWrapper(torch.nn.Module):
    """Wrap YOLO model to export the first output tensor only."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, augment=False)[0]


def _export_model_to_onnx(model, example_input, onnx_path):
    _ensure_parent(onnx_path)
    wrapped = _ExportWrapper(model)
    wrapped.eval()

    kwargs = dict(
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['predictions'],
        dynamic_axes={'images': {0: 'batch_size'}, 'predictions': {0: 'batch_size'}},
    )

    try:
        torch.onnx.export(
            wrapped,
            example_input,
            onnx_path,
            external_data=True,
            **kwargs,
        )
    except TypeError:
        # Older PyTorch versions may not support the external_data argument.
        torch.onnx.export(
            wrapped,
            example_input,
            onnx_path,
            **kwargs,
        )


def _run_pytorch_raw(detector, batch_tensor):
    with torch.no_grad():
        pred = detector.model(batch_tensor, augment=False)[0]
    return pred.cpu().numpy()


def _run_onnx_raw(onnx_path, batch_tensor):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_inputs = {input_name: batch_tensor.cpu().numpy()}
    pred = session.run([output_name], onnx_inputs)[0]
    return pred


def _iou_xywh(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _postprocess(pred_np, image_info, detector, detection_threshold):
    pred_tensor = torch.from_numpy(pred_np)
    nms_iou_thres = 0.45 if 'classic' in detector.compatibility_mode else 0.6
    use_library_nms = (
        (pytorch_detector.yolo_model_type_imported is not None)
        and (pytorch_detector.yolo_model_type_imported == 'ultralytics')
    )
    if use_library_nms:
        pred = pytorch_detector.non_max_suppression(
            prediction=pred_tensor,
            conf_thres=detection_threshold,
            iou_thres=nms_iou_thres,
            agnostic=False,
            multi_label=False,
        )
    else:
        pred = pytorch_detector.nms(
            prediction=pred_tensor,
            conf_thres=detection_threshold,
            iou_thres=nms_iou_thres,
        )

    det = pred[0]
    detections = []

    if len(det) == 0:
        return detections

    scaling_shape = image_info['scaling_shape']
    letterbox_pad = image_info['letterbox_pad']
    img_original = image_info['img_original']

    gn = torch.tensor(scaling_shape)[[1, 0, 1, 0]]

    if 'classic' in detector.compatibility_mode:
        pytorch_detector.scale_coords(image_info['img_processed'].shape[:2], det[:, :4], img_original.shape).round()
    else:
        ratio = (img_original.shape[0] / scaling_shape[0], img_original.shape[1] / scaling_shape[1])
        ratio_pad = (ratio, letterbox_pad)
        pytorch_detector.scale_coords(
            image_info['img_processed'].shape[:2], det[:, :4], scaling_shape, ratio_pad
        ).round()

    for *xyxy, conf, cls in reversed(det):
        if conf < detection_threshold:
            continue

        xywh = (pytorch_detector.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        api_box = ct_utils.convert_yolo_to_xywh(xywh)

        if 'classic' in detector.compatibility_mode:
            api_box = ct_utils.truncate_float_array(api_box, precision=4)
            conf = ct_utils.truncate_float(conf.tolist(), precision=3)
        else:
            api_box = ct_utils.round_float_array(api_box, precision=4)
            conf = ct_utils.round_float(conf.tolist(), precision=3)

        if not detector.use_model_native_classes:
            cls = int(cls.tolist()) + 1
        else:
            cls = int(cls.tolist())

        detections.append({'category': str(cls), 'conf': conf, 'bbox': api_box})

    return detections


def _compare(pt_dets, onnx_dets, iou_threshold):
    pt_remaining = list(pt_dets)
    onnx_remaining = list(onnx_dets)
    matches = []

    while pt_remaining and onnx_remaining:
        best = None
        for i, p in enumerate(pt_remaining):
            for j, o in enumerate(onnx_remaining):
                if p['category'] != o['category']:
                    continue
                iou = _iou_xywh(p['bbox'], o['bbox'])
                if best is None or iou > best[0]:
                    best = (iou, i, j)
        if best is None:
            break
        iou, i, j = best
        if iou < iou_threshold:
            break
        p = pt_remaining.pop(i)
        o = onnx_remaining.pop(j)
        matches.append((p, o, iou))

    bbox_l1 = []
    conf_diffs = []
    ious = []

    for p, o, iou in matches:
        bbox_l1.append(sum(abs(a - b) for a, b in zip(p['bbox'], o['bbox'])))
        conf_diffs.append(abs(float(p['conf']) - float(o['conf'])))
        ious.append(iou)

    def _compute_stat_or_default(values, fn, default=0.0):
        if not values:
            return default
        return float(fn(values))

    is_close = (
        len(pt_remaining) == 0
        and len(onnx_remaining) == 0
        and _compute_stat_or_default(ious, min, 0.0) >= iou_threshold
        and _compute_stat_or_default(conf_diffs, max, 0.0) <= 0.02
    )

    return {
        'matched_pairs': len(matches),
        'pt_unmatched': len(pt_remaining),
        'onnx_unmatched': len(onnx_remaining),
        'max_bbox_l1': _compute_stat_or_default(bbox_l1, max),
        'mean_bbox_l1': _compute_stat_or_default(bbox_l1, np.mean),
        'max_conf_abs_diff': _compute_stat_or_default(conf_diffs, max),
        'mean_conf_abs_diff': _compute_stat_or_default(conf_diffs, np.mean),
        'min_iou': _compute_stat_or_default(ious, min),
        'mean_iou': _compute_stat_or_default(ious, np.mean),
        'is_close': is_close,
    }


def _validate_model(model_name, output_dir, image_path, detection_threshold, iou_threshold):
    print(f'=== {model_name}: downloading model ===')
    model_path = try_download_known_detector(model_name, force_download=False)
    print(f'Model path: {model_path}')

    detector = pytorch_detector.PTDetector(
        model_path,
        detector_options={'force_cpu': True, 'device': 'cpu'},
    )

    print(f'=== {model_name}: preprocessing image {image_path} ===')
    image = Image.open(image_path).convert('RGB')
    image_info = detector.preprocess_image(image, image_id=image_path, image_size=None)

    img = image_info['img_processed']
    batch_tensor = torch.from_numpy(np.ascontiguousarray(img.transpose((2, 0, 1)))).unsqueeze(0).float() / 255.0

    onnx_path = os.path.join(output_dir, f'md_{model_name}.onnx')
    print(f'=== {model_name}: running PT and ONNX inference ===')
    pt_raw = _run_pytorch_raw(detector, batch_tensor)

    print(f'=== {model_name}: exporting ONNX to {onnx_path} ===')
    _export_model_to_onnx(detector.model, batch_tensor, onnx_path)

    onnx_raw = _run_onnx_raw(onnx_path, batch_tensor)

    pt_dets = _postprocess(pt_raw, image_info, detector, detection_threshold)
    onnx_dets = _postprocess(onnx_raw, image_info, detector, detection_threshold)

    comparison = _compare(pt_dets, onnx_dets, iou_threshold=iou_threshold)

    summary = ValidationSummary(
        model=model_name,
        pt_model_path=model_path,
        onnx_model_path=onnx_path,
        pt_detection_count=len(pt_dets),
        onnx_detection_count=len(onnx_dets),
        matched_pairs=comparison['matched_pairs'],
        max_bbox_l1=comparison['max_bbox_l1'],
        mean_bbox_l1=comparison['mean_bbox_l1'],
        max_conf_abs_diff=comparison['max_conf_abs_diff'],
        mean_conf_abs_diff=comparison['mean_conf_abs_diff'],
        min_iou=comparison['min_iou'],
        mean_iou=comparison['mean_iou'],
        is_close=comparison['is_close'],
    )

    print(f'=== {model_name}: validation summary ===')
    print(json.dumps(asdict(summary), indent=2))
    print(f'PT detections: {json.dumps(pt_dets, indent=2)}')
    print(f'ONNX detections: {json.dumps(onnx_dets, indent=2)}')

    return asdict(summary)


def _parse_args():
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--image-path', default='images/idaho-camera-traps.jpg')
    parser.add_argument('--detection-threshold', type=float, default=0.2)
    parser.add_argument('--iou-threshold', type=float, default=0.9)
    parser.add_argument('--strict', action='store_true')
    return parser.parse_args()


def main():
    """Export and validate both target MDv1000 models."""

    args = _parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    image_path = args.image_path
    if not os.path.isabs(image_path):
        image_path = os.path.join(repo_root, image_path)

    os.makedirs(args.output_dir, exist_ok=True)

    models = ['mdv1000-redwood', 'mdv1000-sorrel']
    summaries = []
    had_error = False

    for model_name in models:
        try:
            summary = _validate_model(
                model_name=model_name,
                output_dir=args.output_dir,
                image_path=image_path,
                detection_threshold=args.detection_threshold,
                iou_threshold=args.iou_threshold,
            )
            summaries.append(summary)
        except Exception as e:
            had_error = True
            error_record = {
                'model': model_name,
                'error': str(e),
                'is_close': False,
            }
            summaries.append(error_record)
            print(f'WARNING: failed processing {model_name}: {type(e).__name__}: {e}')
            traceback.print_exc()

    summary_path = os.path.join(args.output_dir, 'validation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2)
    print(f'Wrote validation summary to {summary_path}')

    all_close = all(s.get('is_close', False) for s in summaries)
    if not all_close:
        print('WARNING: ONNX validation is not near-identical for one or more models.')

    if args.strict and (had_error or not all_close):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
