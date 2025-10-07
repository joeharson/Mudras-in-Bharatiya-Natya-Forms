from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from pathlib import Path
import threading
import atexit
import traceback
import time

# Ensure templates directory exists (next to this file)
template_dir = Path(__file__).parent / "templates"
template_dir.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(template_dir))

# Initialize camera (Windows-friendly)
def init_camera(index=0):
    cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print(f"Warning: could not open webcam at index {index}")
        return None
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam

cap = init_camera(0)

def _cleanup():
    try:
        if cap is not None and cap.isOpened():
            cap.release()
    except Exception:
        pass
    cv2.destroyAllWindows()

atexit.register(_cleanup)

# Mediapipe hands (we will keep a global instance but detection uses landmarks)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
# lowered detection/tracking confidences to increase recall for small/distant hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Shared state for frontend
_current_mudra = "No hand"
_state_lock = threading.Lock()

# --- CORRECTED MUDRA MEANINGS ---
MUDRA_INFO = {
    # Updated as per your request
    "Pataka": "Symbolizes a flag, used to denote clouds, forests, or to stop.",
    "Tripataka": "Represents a crown, tree, or arrow.",
    "Ardhapataka": "Half-flag, used for leaves, knives, or a tower.",
    "Kartarimukha": "Represents separation, scissors, or opposition.",
    "Mayura": "The peacock, symbolizing beauty and elegance.",
    "Ardhachandra": "Half-moon, denotes the crescent or hand-seizing.",
    "Arala": "Used to show drinking poison or nectar.",
    "Mushti": "Fist, symbolizing strength or grasping hair.",
    "Shikhara": "Signifies a pillar, bow, or determination.", # Note: Key is Shikhara
    "Kapitha": "Represents Lakshmi or holding cymbals.",
    "Katakamukha": "Signifies a garland, plucking a flower, or holding a necklace.", # Note: Key is Katakamukha
    "Suchi": "Represents pointing to one or the number one.", # Note: Key is Suchi
    "Chandrakala": "Moon digit, shows the moon or a spear.",
    "Mrigashirsha": "Deer head, used to denote women, flute, or caressing.",
    "Simhamukha": "Lion face, used for corals, pearls, or heroic acts.", # Note: Key is Simhamukha
    "Alapadma": "Lotus, symbolizing beauty or offering flowers.", # Note: Key is Alapadma

    # Original meanings for other mudras
    "Shukatunda": "Shukatunda — Parrot's beak: Thumb and index touching, others extended. For birds or shooting arrow.",
    "Padmakosha": "Padmakosha — Lotus bud: Fingers curved like a bud; used for offering or flower motifs.",
    "Sarpashirsha": "Sarpashirsha — Snake head: Some fingers bent to resemble a snake head; expresses winding movement.",
    "Kangula": "Kangula — Little-finger: Only little finger extended; used to mimic small bell or tinkling.",
    "Hamsasya": "Hamsasya — Swan's beak: Pinching with thumb+index+middle; used for delicate picks or swan imagery.",
    "Hamsapaksha": "Hamsapaksha — Swan's wing: Two fingers extended, others bent; suggests flight or movement.",
    "Mukula": "Mukula — Flower bud: Multiple fingertips touching thumb; small bud or delicate object.",
    "Tamrachuda": "Tamrachuda — Rooster's crest: Vertical stacking of finger tips resembling a crest; bird motif.",
    "Trishula": "Trishula — Trident: Three prongs formed by fingers; symbolizes power, weapon or triad.",
    "Ardhasuchi": "Ardhasuchi — Half needle: Partial pointing gesture, for fine or partial indication.",
    "Vardhamana": "Vardhamana — Growing: Gesture indicating growth, prosperity or increase.",
    "Anjali": "Anjali — Prayer (two hands): Palms together; greeting, respect, offering.",
    "Pushpanjali": "Pushpanjali — Flower offering (two hands): Both hands presenting flowers; devotion.",
    "Namaskara": "Namaskara — Salutation (two hands): Greeting with respect.",
    "Dvandva": "Dvandva — Both Pataka: Twin pose representing opposition, conflict or battle.",
    "Chakra": "Chakra — Wheel (two hands): Circular formation denoting movement, cycle or time.",
    "Garuda": "Garuda — Eagle (two hands): Wing-like pose for bird, flight, freedom.",
    "Padma": "Padma — Lotus (two hands): Both hands form a lotus; purity and divinity.",
    "Samyukta Hasta": "Samyukta Hasta — Joined gestures: Combined hand expressions for complex storytelling.",
    "Vajra": "Vajra — Thunderbolt (two hands): Power, strength or divine weapon.",
    "Abhaya": "Abhaya — Fearlessness (two hands): Gesture of protection and blessing.",
    "Unknown": "Unknown — Pose not confidently recognized. Adjust hand position, lighting or camera distance.",
    "No hand": "No hand detected. Place your hand clearly in front of the camera."
}

# Helper detection functions adapted from reference
def _pt(lm):
    return np.array([lm.x, lm.y])

def get_distance(landmarks, p1_id, p2_id):
    return np.linalg.norm(_pt(landmarks[p1_id]) - _pt(landmarks[p2_id]))

def get_hand_scale(landmarks):
    # hand bbox size in normalized coords (robust to scale)
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return max(w, h, 1e-6)

def is_finger_touching(landmarks, tip1, tip2, hand_scale=None, base_ratio=0.35):
    # base_ratio determines how close two tips must be relative to hand size
    if hand_scale is None:
        hand_scale = get_hand_scale(landmarks)
    thresh = hand_scale * base_ratio
    return get_distance(landmarks, tip1, tip2) < thresh

def is_finger_extended(landmarks, tip_id, pip_id, wrist_id=0, margin=1.03):
    # compare distances to wrist: extended if tip is further from wrist than pip by margin factor
    try:
        d_tip = get_distance(landmarks, tip_id, wrist_id)
        d_pip = get_distance(landmarks, pip_id, wrist_id)
        return d_tip > d_pip * margin
    except Exception:
        return False

def is_finger_bent(landmarks, tip_id, pip_id, wrist_id=0, margin=0.98):
    try:
        d_tip = get_distance(landmarks, tip_id, wrist_id)
        d_pip = get_distance(landmarks, pip_id, wrist_id)
        return d_tip < d_pip * margin
    except Exception:
        return False

def detect_single_hand_mudra(landmarks, image_height, image_width):
    # landmarks: sequence of normalized landmark objects (landmark.x/landmark.y)
    try:
        # compute a hand scale once and use scale-aware heuristics
        hand_scale = get_hand_scale(landmarks)
        # thumb: compare tip vs IP relative to wrist
        thumb_extended = is_finger_extended(landmarks, 4, 3, wrist_id=0)
        thumb_bent = is_finger_bent(landmarks, 4, 3, wrist_id=0)
    except Exception:
        thumb_extended = thumb_bent = False

    index_extended = is_finger_extended(landmarks, 8, 6)
    index_bent = is_finger_bent(landmarks, 8, 6)
    middle_extended = is_finger_extended(landmarks, 12, 10)
    middle_bent = is_finger_bent(landmarks, 12, 10)
    ring_extended = is_finger_extended(landmarks, 16, 14)
    ring_bent = is_finger_bent(landmarks, 16, 14)
    pinky_extended = is_finger_extended(landmarks, 20, 18)
    pinky_bent = is_finger_bent(landmarks, 20, 18)

    fingers_extended = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
    fingers_bent = [thumb_bent, index_bent, middle_bent, ring_bent, pinky_bent]

    # Detection rules (adapted and ordered)
    if all([index_extended, middle_extended, ring_extended, pinky_extended]) and thumb_bent:
        return "Pataka", (0, 255, 0), MUDRA_INFO["Pataka"]
    if all([index_extended, middle_extended, ring_extended]) and pinky_bent and thumb_bent:
        return "Tripataka", (0,255,0), MUDRA_INFO["Tripataka"]
    if all([index_extended, middle_extended]) and all([ring_bent, pinky_bent, thumb_bent]):
        return "Ardhapataka", (0,255,0), MUDRA_INFO["Ardhapataka"]
    if all([index_bent, middle_bent]) and all([ring_extended, pinky_extended]) and thumb_bent:
        return "Kartarimukha", (0,255,0), MUDRA_INFO["Kartarimukha"]
    if all([index_bent, middle_bent, ring_bent, pinky_bent]) and thumb_extended:
        return "Mayura", (0,255,0), MUDRA_INFO["Mayura"]
    if all([index_extended, middle_extended, ring_extended, pinky_extended]) and (not thumb_extended) and (not thumb_bent):
        return "Ardhachandra", (0,255,0), MUDRA_INFO["Ardhachandra"]
    if index_bent and all([middle_extended, ring_extended, pinky_extended]) and thumb_bent:
        return "Arala", (0,255,0), MUDRA_INFO["Arala"]
    if is_finger_touching(landmarks, 4, 8) and all([middle_extended, ring_extended, pinky_extended]):
        return "Shukatunda", (0,255,0), MUDRA_INFO.get("Shukatunda", "Shukatunda — Parrot's beak")
    if all(fingers_bent):
        return "Mushti", (0,255,0), MUDRA_INFO["Mushti"]
    if thumb_extended and all([index_bent, middle_bent, ring_bent, pinky_bent]):
        return "Shikhara", (0,255,0), MUDRA_INFO["Shikhara"]
    if is_finger_touching(landmarks, 4, 8) and all([middle_extended, ring_extended, pinky_extended]):
        return "Kapitha", (0,255,0), MUDRA_INFO["Kapitha"]
    if is_finger_touching(landmarks, 4, 8) and is_finger_touching(landmarks, 4, 12) and all([ring_extended, pinky_extended]):
        return "Katakamukha", (0,255,0), MUDRA_INFO["Katakamukha"]
    if index_extended and all([not thumb_extended and index_extended, not middle_extended and not ring_extended]) and all([thumb_bent, middle_bent, ring_bent, pinky_bent]):
        # fallthrough simpler check
        pass
    if index_extended and all([thumb_bent, middle_bent, ring_bent, pinky_bent]):
        return "Suchi", (0,255,0), MUDRA_INFO["Suchi"]
    if is_finger_touching(landmarks, 4, 8) and all([middle_extended, ring_extended, pinky_extended]) and get_distance(landmarks,4,8) < 0.06:
        return "Chandrakala", (0,255,0), MUDRA_INFO["Chandrakala"]
    if all(fingers_bent) and thumb_bent:
        return "Padmakosha", (0,255,0), MUDRA_INFO.get("Padmakosha","Padmakosha — Lotus bud")
    if all([index_bent, middle_bent]) and all([ring_extended, pinky_extended]) and thumb_bent:
        return "Sarpashirsha", (0,255,0), MUDRA_INFO.get("Sarpashirsha","Sarpashirsha — Snake head")
    if all([index_extended, middle_extended]) and all([not ring_extended, not pinky_extended, not thumb_extended]) and abs(landmarks[8].y - landmarks[12].y) < 0.03:
        return "Mrigashirsha", (0,255,0), MUDRA_INFO["Mrigashirsha"]
    if all([index_extended, middle_extended, ring_extended]) and pinky_bent and thumb_bent:
        return "Simhamukha", (0,255,0), MUDRA_INFO["Simhamukha"]
    if pinky_extended and all([thumb_bent, index_bent, middle_bent, ring_bent]):
        return "Kangula", (0,255,0), MUDRA_INFO.get("Kangula","Kangula — Little finger")
    if all(fingers_extended):
        return "Alapadma", (0,255,0), MUDRA_INFO.get("Alapadma","Alapadma — Fully bloomed lotus")
    if is_finger_touching(landmarks, 4, 8) and is_finger_touching(landmarks, 4, 12) and all([ring_bent, pinky_bent]):
        return "Hamsasya", (0,255,0), MUDRA_INFO.get("Hamsasya","Hamsasya — Swan's beak")
    if all([index_extended, middle_extended]) and all([ring_bent, pinky_bent, thumb_bent]) and get_distance(landmarks,8,12) > 0.05:
        return "Hamsapaksha", (0,255,0), MUDRA_INFO.get("Hamsapaksha","Hamsapaksha — Swan's wing")
    if is_finger_touching(landmarks, 4, 8) and is_finger_touching(landmarks, 4, 12) and is_finger_touching(landmarks, 4, 16) and is_finger_touching(landmarks, 4, 20):
        return "Mukula", (0,255,0), MUDRA_INFO.get("Mukula","Mukula — Flower bud")
    if all([index_extended, middle_extended, ring_extended]) and pinky_bent and thumb_bent and landmarks[8].y < landmarks[12].y < landmarks[16].y:
        return "Tamrachuda", (0,255,0), MUDRA_INFO.get("Tamrachuda","Tamrachuda — Rooster's crest")
    if all([index_extended, middle_extended, ring_extended]) and pinky_bent and thumb_bent and abs(landmarks[8].y - landmarks[12].y) < 0.02 and abs(landmarks[12].y - landmarks[16].y) < 0.02:
        return "Trishula", (0,255,0), MUDRA_INFO.get("Trishula","Trishula — Trident")
    if index_extended and middle_bent and all([ring_bent, pinky_bent, thumb_bent]):
        return "Ardhasuchi", (0,255,0), MUDRA_INFO.get("Ardhasuchi","Ardhasuchi — Half needle")
    if is_finger_touching(landmarks, 4, 8) and all([middle_extended, ring_extended, pinky_extended]):
        return "Vardhamana", (0,255,0), MUDRA_INFO.get("Vardhamana","Vardhamana — Growing")
    return "", (128,128,128), ""

def detect_two_hand_mudra(left_landmarks, right_landmarks, image_height, image_width):
    # helper local wrappers
    def lf(tip, pip): return is_finger_extended(left_landmarks, tip, pip)
    def rf(tip, pip): return is_finger_extended(right_landmarks, tip, pip)
    def ltouch(a,b): return is_finger_touching(left_landmarks, a, b)
    def rtouch(a,b): return is_finger_touching(right_landmarks, a, b)
    def get_dist_hands(): return np.sqrt((left_landmarks[9].x - right_landmarks[9].x)**2 + (left_landmarks[9].y - right_landmarks[9].y)**2)

    left_thumb_ext = left_landmarks[4].x > left_landmarks[3].x + 0.02
    left_thumb_bent = left_landmarks[4].x < left_landmarks[3].x - 0.02
    left_index_ext = lf(8,6)
    left_middle_ext = lf(12,10)
    left_ring_ext = lf(16,14)
    left_pinky_ext = lf(20,18)

    right_thumb_ext = right_landmarks[4].x > right_landmarks[3].x + 0.02
    right_thumb_bent = right_landmarks[4].x < right_landmarks[3].x - 0.02
    right_index_ext = rf(8,6)
    right_middle_ext = rf(12,10)
    right_ring_ext = rf(16,14)
    right_pinky_ext = rf(20,18)

    hand_distance = get_dist_hands()

    if (ltouch(4,8) and rtouch(4,8) and all([left_middle_ext,left_ring_ext,left_pinky_ext]) and all([right_middle_ext,right_ring_ext,right_pinky_ext]) and hand_distance < 0.2):
        return "Anjali", (255,0,0), MUDRA_INFO["Anjali"]
    if (ltouch(4,8) and rtouch(4,8) and all([left_middle_ext,left_ring_ext,left_pinky_ext]) and all([right_middle_ext,right_ring_ext,right_pinky_ext]) and hand_distance < 0.15):
        return "Pushpanjali", (255,0,0), MUDRA_INFO["Pushpanjali"]
    if (all([left_index_ext,left_middle_ext,left_ring_ext,left_pinky_ext]) and left_thumb_bent and all([right_index_ext,right_middle_ext,right_ring_ext,right_pinky_ext]) and right_thumb_bent and hand_distance < 0.25):
        return "Namaskara", (255,0,0), MUDRA_INFO["Namaskara"]
    if (all([left_index_ext,left_middle_ext,left_ring_ext,left_pinky_ext]) and left_thumb_bent and all([right_index_ext,right_middle_ext,right_ring_ext,right_pinky_ext]) and right_thumb_bent and abs(left_landmarks[9].y - right_landmarks[9].y) < 0.08):
        return "Dvandva", (255,0,0), MUDRA_INFO.get("Dvandva","Dvandva — Conflict")
    if (all([left_index_ext,left_middle_ext]) and all([left_thumb_bent,left_ring_ext,left_pinky_ext]) and all([right_index_ext,right_middle_ext]) and all([right_thumb_bent,right_ring_ext,right_pinky_ext]) and hand_distance < 0.3):
        return "Samyukta Ardhapataka", (255,0,0), MUDRA_INFO.get("Samyukta Ardhapataka","Samyukta Ardhapataka — Combined half-flag")
    if (ltouch(4,8) and rtouch(4,8) and all([left_middle_ext,left_ring_ext,left_pinky_ext]) and all([right_middle_ext,right_ring_ext,right_pinky_ext]) and hand_distance < 0.25):
        return "Chakra", (255,0,0), MUDRA_INFO.get("Chakra","Chakra — Wheel")
    if (all([left_index_ext,left_middle_ext,left_ring_ext,left_pinky_ext]) and left_thumb_bent and all([right_index_ext,right_middle_ext,right_ring_ext,right_pinky_ext]) and right_thumb_bent and left_landmarks[9].x < right_landmarks[9].x and hand_distance > 0.2):
        return "Garuda", (255,0,0), MUDRA_INFO.get("Garuda","Garuda — Eagle")
    if (ltouch(4,8) and rtouch(4,8) and all([left_middle_ext,left_ring_ext,left_pinky_ext]) and all([right_middle_ext,right_ring_ext,right_pinky_ext]) and hand_distance < 0.3):
        return "Padma", (255,0,0), MUDRA_INFO.get("Padma","Padma — Lotus")
    if (all([left_index_ext,left_middle_ext]) and all([left_thumb_bent,left_ring_ext,left_pinky_ext]) and all([right_index_ext,right_middle_ext]) and all([right_thumb_bent,right_ring_ext,right_pinky_ext]) and hand_distance < 0.4):
        return "Samyukta Hasta", (255,0,0), MUDRA_INFO.get("Samyukta Hasta","Samyukta Hasta — Combined gestures")
    if (ltouch(4,8) and rtouch(4,8) and all([left_middle_ext,left_ring_ext,left_pinky_ext]) and all([right_middle_ext,right_ring_ext,right_pinky_ext]) and hand_distance < 0.2 and abs(left_landmarks[9].y - right_landmarks[9].y) < 0.05):
        return "Vajra", (255,0,0), MUDRA_INFO.get("Vajra","Vajra — Thunderbolt")
    if (left_thumb_ext and all([left_index_ext,left_middle_ext,left_ring_ext,left_pinky_ext]) and right_thumb_ext and all([right_index_ext,right_middle_ext,right_ring_ext,right_pinky_ext]) and hand_distance > 0.3):
        return "Abhaya", (255,0,0), MUDRA_INFO.get("Abhaya","Abhaya — Fearlessness")
    return "", (128,128,128), ""

def detect_mudra_single_or_two(results, image_height, image_width, mode='Auto'):
    # results.multi_hand_landmarks: list of HandLandmark objects
    if not results or not results.multi_hand_landmarks:
        return "No hand", (128,128,128), MUDRA_INFO["No hand"]

    num_hands = len(results.multi_hand_landmarks)
    if mode == 'Two Hand Only' and num_hands == 2:
        left = results.multi_hand_landmarks[0]
        right = results.multi_hand_landmarks[1]
        # order by x center
        if left.landmark[9].x > right.landmark[9].x:
            left, right = right, left
        return detect_two_hand_mudra(left.landmark, right.landmark, image_height, image_width)

    if mode == 'Single Hand Only' and num_hands >= 1:
        return detect_single_hand_mudra(results.multi_hand_landmarks[0].landmark, image_height, image_width)

    # Auto mode
    if num_hands == 2:
        left = results.multi_hand_landmarks[0]
        right = results.multi_hand_landmarks[1]
        if left.landmark[9].x > right.landmark[9].x:
            left, right = right, left
        two_name, two_col, two_mean = detect_two_hand_mudra(left.landmark, right.landmark, image_height, image_width)
        if two_name:
            return two_name, two_col, two_mean
        # fallback to single detections
        sname, scol, smean = detect_single_hand_mudra(left.landmark, image_height, image_width)
        if sname:
            return sname, scol, smean
        return detect_single_hand_mudra(right.landmark, image_height, image_width)
    else:
        return detect_single_hand_mudra(results.multi_hand_landmarks[0].landmark, image_height, image_width)

# Frame generator using mediapipe results
def generate_frames():
    if cap is None:
        # fallback image
        blank = np.full((360,640,3), 230, dtype=np.uint8)
        cv2.putText(blank, "No camera", (30,180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60,60,60),2)
        ret, buf = cv2.imencode('.jpg', blank)
        fallback = buf.tobytes() if ret else b''
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + fallback + b'\r\n')
            time.sleep(0.2)

    while True:
        try:
            success, frame = cap.read()
            if not success or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            image_height, image_width, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            name, color, meaning = detect_mudra_single_or_two(results, image_height, image_width, mode='Auto')

            # draw landmarks if present
            if results and results.multi_hand_landmarks:
                for hland in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hland, mp_hands.HAND_CONNECTIONS)

            # update state
            with _state_lock:
                global _current_mudra
                _current_mudra = name if name else "Unknown"

            # overlay label
            label = f"{_current_mudra}"
            # remove solid black box; draw outlined text for visibility
            cv2.putText(frame, label, (30,170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 6, cv2.LINE_AA)  # outline
            cv2.putText(frame, label, (30,170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)      # foreground

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception:
            traceback.print_exc()
            time.sleep(0.05)
            continue

# Routes
@app.route('/')
def index():
    # prefer index.html next to this file; if it has Jinja placeholders,
    # replace them with concrete URLs so the browser loads /video_feed correctly.
    def _serve_with_replacements(p: Path):
        txt = p.read_text(encoding='utf-8')
        txt = txt.replace("{{ url_for('video_feed') }}", "/video_feed")
        txt = txt.replace("{{ url_for(\"video_feed\") }}", "/video_feed")
        txt = txt.replace("{{ url_for('status') }}", "/status")
        txt = txt.replace("{{ url_for(\"status\") }}", "/status")
        return txt

    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return _serve_with_replacements(index_path)

    # fallback to project root index.html (one level up)
    root_index = Path(__file__).parent.parent / "index.html"
    if root_index.exists():
        return _serve_with_replacements(root_index)

    # final fallback to Flask templates (if present) — will be rendered by Jinja
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with _state_lock:
        mudra = _current_mudra
    explanation = MUDRA_INFO.get(mudra, MUDRA_INFO["Unknown"])
    # optional example same as explanation
    return jsonify({"mudra": mudra, "explanation": explanation, "example": explanation})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    finally:
        _cleanup()