import cv2
import torch
import numpy as np
import time
import os
import argparse
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from insightface.app import FaceAnalysis

try:
    from boxmot.reid.core import ReID
    from boxmot.trackers.bbox.botsort import BotSort
except ImportError:
    from boxmot.reid.core.reid import ReID
    from boxmot.trackers.botsort.botsort import BotSort

# ============================================
# Initial Settings
# ============================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================
# Configuration
# ============================================
parser = argparse.ArgumentParser(description="Run PPE detection and face recognition on a video")
parser.add_argument("--video", required=True, help="Input video filename or path")
parser.add_argument("--reid", action="store_true", help="Enable ReID")
args = parser.parse_args()

FACE_DETECTION_INTERVAL = 20
PPE_SIZE = 320 if device == "cpu" else 480
FACE_SIZE = 160 if device == "cpu" else 320
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.3
THRESHOLD_FACE = 0.35
USE_HALF = False if device == "cuda:0" else False
TRACKER_ENABLED = True
USE_REID = args.reid
class_names = ["helmet", "head", "body", "vest"]
NUM_CLASSES = len(class_names) + 1
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(PROJECT_DIR, "FaceRecognition", "database")
PPE_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "faster_rcnn_best_stage4_mAP_0.3957.pth")
REID_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "osnet_x0_25_msmt17.pt")
INPUT_VIDEO_DIR = os.path.join(PROJECT_DIR, "input_video")
OUTPUT_VIDEO_DIR = os.path.join(PROJECT_DIR, "output_video")
VIDEO_PATH = args.video if os.path.isabs(args.video) else os.path.join(INPUT_VIDEO_DIR, args.video)
OUTPUT_PATH = os.path.join(OUTPUT_VIDEO_DIR, f"{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_processed.mp4")
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# ============================================
# Load Face Analysis Model
# ============================================
print("Loading face recognition model...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda:0" else ["CPUExecutionProvider"]
)
app.prepare(ctx_id=0 if device == "cuda:0" else -1, det_size=(FACE_SIZE, FACE_SIZE))
print("Face recognition model loaded successfully")

# ============================================
# Load PPE Detection Model
# ============================================
if not os.path.exists(PPE_MODEL_PATH):
    print(f"Model not found: {PPE_MODEL_PATH}")
    exit()
print("Loading PPE detection model...")
model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.load_state_dict(torch.load(PPE_MODEL_PATH, map_location=device))
if USE_HALF:
    model = model.half()
model.to(device)
model.eval()
print("PPE detection model loaded successfully")

# ============================================
# Load Face Database
# ============================================
database = {}
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
if not os.path.exists(DATABASE_DIR):
    print(f"Face database not found: {DATABASE_DIR}")
else:
    for person in os.listdir(DATABASE_DIR):
        person_dir = os.path.join(DATABASE_DIR, person)
        emb_path = os.path.join(person_dir, "embedding.npy")
        if os.path.exists(emb_path):
            emb = np.load(emb_path)
            emb = emb / np.linalg.norm(emb)
            database[person] = emb
            print(f"Loaded face embedding: {person}")

# ============================================
# Helper Functions
# ============================================
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def get_face_name(face_boxes, face_names, ppe_box):
    best_name = "Unknown"
    best_iou = 0
    for face_box, name in zip(face_boxes, face_names):
        iou = calculate_iou(ppe_box, face_box)
        if iou > best_iou and iou > IOU_THRESHOLD:
            best_iou = iou
            best_name = name
            if iou > 0.8:
                break
    return best_name

def get_current_class(track_box, det_boxes, det_labels):
    best_iou = 0
    best_cls = -1
    for box, cls in zip(det_boxes, det_labels):
        iou = calculate_iou(track_box, box)
        if iou > best_iou:
            best_iou = iou
            best_cls = cls
    if best_iou > 0.5:
        return int(best_cls)
    return -1

# ============================================
# Open Video
# ============================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Video not found: {VIDEO_PATH}")
    exit()
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {w}x{h}, FPS: {fps}, Frames: {total_frames}")

# ============================================
# Initialize Video Writer
# ============================================
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

# ============================================
# Initialize BoT-SORT Tracker
# ============================================
if TRACKER_ENABLED:
    print("Initializing BoT-SORT tracker...")
    reid_backend = None
    if USE_REID:
        print("ReID: Enabled")
        print(f"Loading ReID model: {REID_MODEL_PATH}")
        if not os.path.exists(REID_MODEL_PATH):
            print(f"ReID model not found: {REID_MODEL_PATH}")
            exit()
        reid = ReID(REID_MODEL_PATH, device=device, half=False)
        reid_backend = getattr(reid, "model", reid)
        print("ReID model loaded successfully")
    else:
        print("ReID: Disabled")
        print("BoT-SORT will run without ReID")
    tracker = BotSort(
        reid_model=reid_backend,
        with_reid=USE_REID,
        track_buffer=30,
        frame_rate=fps,
        track_high_thresh=CONF_THRESHOLD,
        track_low_thresh=0.1,
        new_track_thresh=CONF_THRESHOLD,
        per_class=True
    )
    print("BoT-SORT tracker initialized")
else:
    tracker = None
    print("Tracker: Disabled")

# ============================================
# Configuration Output
# ============================================
print("-" * 40)
print(f"Tracker: {'Enabled' if TRACKER_ENABLED else 'Disabled'}")
print(f"ReID: {'Enabled' if USE_REID else 'Disabled'}")
print(f"Face detection interval: {FACE_DETECTION_INTERVAL}")
print(f"PPE input size: {PPE_SIZE}x{PPE_SIZE}")
print(f"Face input size: {FACE_SIZE}x{FACE_SIZE}")
print("Track buffer: 30 frames")
print("-" * 40)

# ============================================
# Frame Processing Variables
# ============================================
start_time = time.time()
frame_count = 0
face_boxes = []
face_names = []
last_face_frame = -FACE_DETECTION_INTERVAL
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
if USE_HALF:
    mean = mean.half()
    std = std.half()
mean = mean.to(device)
std = std.to(device)

# ============================================
# Main Processing Loop
# ============================================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # ========================================
    # Face Detection
    # ========================================
    if (frame_count - last_face_frame) >= FACE_DETECTION_INTERVAL:
        last_face_frame = frame_count
        faces = app.get(frame)
        face_boxes = []
        face_names = []
        for face in faces:
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)
            best_name = "Unknown"
            best_score = -1
            for person_name, db_emb in database.items():
                score = cosine_similarity(emb, db_emb)
                if score > best_score:
                    best_score = score
                    best_name = person_name
                    if score > 0.9:
                        break
            if best_score < THRESHOLD_FACE:
                best_name = "Unknown"
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_boxes.append([x1, y1, x2, y2])
            face_names.append(best_name)
    # ========================================
    # PPE Detection
    # ========================================
    img_resized = cv2.resize(frame, (PPE_SIZE, PPE_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to(device)
    img_tensor = ((img_tensor - mean) / std).unsqueeze(0)
    if USE_HALF:
        img_tensor = img_tensor.half()
    with torch.no_grad():
        pred = model(img_tensor)
    boxes = pred[0]["boxes"].cpu().numpy()
    scores = pred[0]["scores"].cpu().numpy()
    labels = pred[0]["labels"].cpu().numpy()
    mask = scores >= CONF_THRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    labels = labels - 1
    scale_x = w / PPE_SIZE
    scale_y = h / PPE_SIZE
    if len(boxes) > 0:
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        boxes = boxes.astype(np.int32)
    # ========================================
    # Update BoT-SORT
    # ========================================
    if TRACKER_ENABLED:
        if len(boxes) > 0:
            detections = np.column_stack([boxes, scores, labels]).astype(np.float32)
        else:
            detections = np.empty((0, 6), dtype=np.float32)
        tracks = tracker.update(detections, frame)
    else:
        tracks = np.empty((0, 7))
    # ========================================
    # Draw Results
    # ========================================
    result = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        conf = float(track[5])
        track_box = [x1, y1, x2, y2]
        cls = -1
        if len(track) > 6:
            cls = int(track[6])
        if cls < 0 or cls > 3:
            cls = get_current_class(track_box, boxes, labels)
        if cls == -1:
            continue
        if cls in [0, 1]:
            person_name = get_face_name(face_boxes, face_names, track_box)
            if cls == 0:
                color = (0, 255, 255)
                label = "Helmet"
            else:
                color = (0, 165, 255)
                label = "Head"
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
            cv2.putText(result, f"{person_name} {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
            cv2.putText(result, f"ID:{track_id} C:{conf:.2f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        else:
            if cls == 2:
                color = (0, 200, 0)
                label = "Body"
            else:
                color = (0, 0, 255)
                label = "Vest"
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
            cv2.putText(result, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
    # ========================================
    # Write Output
    # ========================================
    out.write(result)
    if frame_count % 100 == 0:
        elapsed = time.time() - start_time
        fps_actual = frame_count / elapsed if elapsed > 0 else 0
        print(f"Frame: {frame_count}/{total_frames} | FPS: {fps_actual:.1f}")

# ============================================
# Finish
# ============================================
cap.release()
out.release()
elapsed_time = time.time() - start_time
avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
print("-" * 40)
print("Processing completed")
print(f"Time: {elapsed_time:.1f}s")
print(f"Frames processed: {frame_count}")
print(f"Average FPS: {avg_fps:.1f}")
print(f"ReID: {'Enabled' if USE_REID else 'Disabled'}")
print(f"Output saved: {OUTPUT_PATH}")
print("-" * 40)