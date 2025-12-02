# gesture_ml.py
# Modes: collect, train, run
# Usage:
#   python gesture_ml.py collect
#   python gesture_ml.py train
#   python gesture_ml.py run

import os, sys, time, math, csv
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from ctypes import cast, POINTER
import ctypes

# ML libs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

# Audio/media control (Windows)
from comtypes import CLSCTX_ALL, CoInitialize
from comtypes.client import CreateObject
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ---------- Config ----------
DATA_CSV = "gesture_data.csv"
MODEL_FILE = "gesture_model.joblib"

PINCH_MIN = 40
PINCH_MAX = 240
PINCH_MUTE_THRESHOLD = 30

# require same label for this many seconds to fire
HOLD_SECONDS = 2.0

# Buttons mapping for collection
LABEL_KEYS = {
    ord('p'): "play_pause",
    ord('n'): "next",
    ord('r'): "previous",
    ord('0'): "none"
}

# ---------- Helpers: Media keys ----------
VK_MEDIA_NEXT = 0xB0
VK_MEDIA_PREV = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3
_user32 = ctypes.windll.user32
def press_media_key(vk):
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002
    _user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
    time.sleep(0.03)
    _user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

# ---------- Helpers: pycaw volume ----------
def get_volume_interface():
    try:
        dev = AudioUtilities.GetSpeakers()
        if hasattr(dev, "Activate"):
            iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol = cast(iface, POINTER(IAudioEndpointVolume))
            return vol
    except Exception:
        pass
    try:
        CoInitialize()
        enumerator = CreateObject("MMDeviceEnumerator.MMDeviceEnumerator")
        device = enumerator.GetDefaultAudioEndpoint(0, 1)
        iface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        vol = cast(iface, POINTER(IAudioEndpointVolume))
        return vol
    except Exception:
        pass
    return None

volume_interface = get_volume_interface()
if volume_interface is None:
    VOL_RANGE = (-65.25, 0.0)
else:
    try:
        vr = volume_interface.GetVolumeRange()
        VOL_RANGE = (float(vr[0]), float(vr[1]))
    except Exception:
        VOL_RANGE = (-65.25, 0.0)

# ---------- Mediapipe init ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.6)

def landmark_features(landmarks, img_w, img_h):
    """
    Return normalized feature vector from mediapipe landmarks for single hand.
    Normalize coordinates relative to wrist (landmark 0) and by hand width (dist between 5 and 17).
    Output: flat array of length 42 (x,y per 21 landmarks) but omit z.
    """
    lm = [[int(p.x*img_w), int(p.y*img_h)] for p in landmarks]
    wrist = lm[0]
    hand_w = math.hypot(lm[5][0]-lm[17][0], lm[5][1]-lm[17][1]) + 1e-6
    feat = []
    for x,y in lm:
        fx = (x - wrist[0]) / hand_w
        fy = (y - wrist[1]) / hand_w
        feat.extend([fx, fy])
    return feat

# ---------- Data collection ----------
def collect_data():
    print("COLLECT MODE: Press keys to label frames:")
    print(" p = play_pause, n = next, r = previous, 0 = none, q = quit")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    rows_collected = 0
    # create file with header if doesn't exist
    header = ["f"+str(i) for i in range(42)] + ["label"]
    if not os.path.exists(DATA_CSV):
        with open(DATA_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h,w,_ = frame.shape
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img)
        display = frame.copy()
        label = None
        cv2.putText(display, f"Collected: {rows_collected}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)
            feats = landmark_features(hand.landmark, w, h)
        else:
            feats = None
        cv2.imshow("Collect (press p/n/r/0)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in LABEL_KEYS:
            lab = LABEL_KEYS[key]
            if feats is not None:
                row = feats + [lab]
                with open(DATA_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                rows_collected += 1
                print(f"[COLLECT] saved label={lab} (total {rows_collected})")
            else:
                print("[WARN] No hand detected - not saved")
    cap.release()
    cv2.destroyAllWindows()
    print("Collection finished.")

# ---------- Training ----------
def train_model():
    if not os.path.exists(DATA_CSV):
        print("No data file found. Run 'collect' first.")
        return
    print("Loading CSV...", DATA_CSV)
    X=[]
    y=[]
    with open(DATA_CSV, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            feats = list(map(float, row[:-1]))
            label = row[-1]
            X.append(feats)
            y.append(label)
    X = np.array(X); y = np.array(y)
    print("Data shape:", X.shape, y.shape, "labels:", set(y))
    # simple classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    print("Cross-validating (5-fold)...")
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("CV acc:", scores.mean(), "±", scores.std())
    print("Training on full dataset...")
    clf.fit(X, y)
    joblib.dump(clf, MODEL_FILE)
    print("Saved model to", MODEL_FILE)

# ---------- Runtime inference ----------
def run_inference():
    if not os.path.exists(MODEL_FILE):
        print("No model found. Run 'train' first.")
        return
    clf = joblib.load(MODEL_FILE)
    print("Loaded model. Running inference. Hold same prediction for", HOLD_SECONDS, "sec to trigger.")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    last_pred = None
    pred_start_time = None
    prev_t = time.time()
    fps = 0.0
    # To reduce accidental triggers, require a minimum confidence
    # (RandomForest predict_proba available). We'll require prob >= 0.6
    MIN_PROB = 0.6
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h,w,_ = frame.shape
        cur_t = time.time()
        fps = 0.9*fps + 0.1*(1.0/max(1e-6, cur_t-prev_t))
        prev_t = cur_t

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img)
        display = frame.copy()
        candidate = None
        candidate_prob = 0.0
        pinch = None

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(display, hand, mp_hands.HAND_CONNECTIONS)
            feats = landmark_features(hand.landmark, w, h)
            X = np.array(feats).reshape(1,-1)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)[0]
                labels = clf.classes_
                idx = np.argmax(probs)
                candidate = labels[idx]
                candidate_prob = probs[idx]
            else:
                candidate = clf.predict(X)[0]
                candidate_prob = 1.0
            # compute pinch distance using tip landmarks (index=8, thumb=4)
            tx = int(hand.landmark[4].x * w); ty = int(hand.landmark[4].y * h)
            ix = int(hand.landmark[8].x * w); iy = int(hand.landmark[8].y * h)
            pinch = math.hypot(tx-ix, ty-iy)
        else:
            candidate = None
            candidate_prob = 0.0

        # Immediate volume mode if pinch in range
        if pinch is not None and PINCH_MIN <= pinch <= PINCH_MAX:
            vol_scalar = float(np.interp(pinch, [PINCH_MIN, PINCH_MAX], [0.0, 1.0]))
            vol_db = float(np.interp(pinch, [PINCH_MIN, PINCH_MAX], [VOL_RANGE[0], VOL_RANGE[1]]))
            mute = pinch <= PINCH_MUTE_THRESHOLD
            # try master volume
            if volume_interface is not None:
                try:
                    volume_interface.SetMasterVolumeLevel(vol_db, None)
                    volume_interface.SetMute(1 if mute else 0, None)
                except Exception:
                    pass
            # otherwise fallback: set per-app sessions (already handled by previous code in your repo if needed)
            # we do NOT print volume (per user's request)
            # reset candidate timers so media doesn't trigger while pinching
            last_pred = None
            pred_start_time = None

        else:
            # classify only if probability high enough (avoid weak votes)
            if candidate is not None and candidate != "none" and candidate_prob >= MIN_PROB:
                # same-prediction required
                if last_pred != candidate:
                    last_pred = candidate
                    pred_start_time = cur_t
                else:
                    # same candidate continuing
                    elapsed = cur_t - (pred_start_time or cur_t)
                    # show a progress bar for hold
                    pct = min(1.0, elapsed / HOLD_SECONDS)
                    # Draw progress bar
                    x0,y0,bw,bh = 20,140,300,22
                    cv2.rectangle(display, (x0,y0), (x0+bw, y0+bh), (50,50,50), -1)
                    fill = int(bw * pct)
                    cv2.rectangle(display, (x0,y0), (x0+fill, y0+bh), (0,200,0), -1)
                    cv2.rectangle(display, (x0,y0), (x0+bw, y0+bh), (200,200,200), 2)
                    cv2.putText(display, f"Holding: {last_pred} {int(pct*100)}%", (x0, y0-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
                    if elapsed >= HOLD_SECONDS:
                        # Trigger action and reset
                        if last_pred == "play_pause":
                            press_media_key(VK_MEDIA_PLAY_PAUSE)
                            print(f"[ACTION] PLAY/PAUSE triggered at {time.strftime('%H:%M:%S')}")
                        elif last_pred == "next":
                            press_media_key(VK_MEDIA_NEXT)
                            print(f"[ACTION] NEXT TRACK triggered at {time.strftime('%H:%M:%S')}")
                        elif last_pred == "previous":
                            press_media_key(VK_MEDIA_PREV)
                            print(f"[ACTION] PREVIOUS TRACK triggered at {time.strftime('%H:%M:%S')}")
                        # reset
                        last_pred = None
                        pred_start_time = None
                        # small cooldown to avoid re-trigger immediately
                        time.sleep(0.5)
            else:
                # either no candidate or low prob => reset timer
                last_pred = None
                pred_start_time = None

        # overlays
        cv2.putText(display, f"FPS:{int(fps)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        if pinch is not None:
            cv2.putText(display, f"pinch:{int(pinch)}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        if candidate is not None:
            cv2.putText(display, f"pred:{candidate} p={candidate_prob:.2f}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow("Gesture (ML) Run", display)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- CLI ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python gesture_ml.py [collect|train|run]")
        return
    mode = sys.argv[1].lower()
    if mode == "collect":
        collect_data()
    elif mode == "train":
        train_model()
    elif mode == "run":
        run_inference()
    else:
        print("Unknown mode:", mode)

if __name__ == "__main__":
    main()
