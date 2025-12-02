import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, CoInitialize
from comtypes.client import CreateObject
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# --------- ALWAYS-DEFINED fallback values (avoid NameError) ----------
FALLBACK_MIN_VOL = -65.25
FALLBACK_MAX_VOL = 0.0
minVol = FALLBACK_MIN_VOL
maxVol = FALLBACK_MAX_VOL


# --------- Function to get master volume interface ----------
def get_master_volume_interface():
    # 1) Normal pycaw route
    try:
        dev = AudioUtilities.GetSpeakers()
        if hasattr(dev, "Activate"):
            interface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            return volume, "pycaw:GetSpeakers()"
    except:
        pass

    # 2) Direct COM enumerator (most reliable)
    try:
        CoInitialize()
        enumerator = CreateObject("MMDeviceEnumerator.MMDeviceEnumerator")
        # eRender=0, eMultimedia=1
        device = enumerator.GetDefaultAudioEndpoint(0, 1)
        interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        return volume, "comtypes:MMDeviceEnumerator"
    except:
        pass

    # 3) Try first device fallback
    try:
        devices = AudioUtilities.GetAllDevices()
        for d in devices:
            if hasattr(d, "Activate"):
                interface = d.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
                return volume, "pycaw:GetAllDevices()"
    except:
        pass

    return None, "none"


# --------- Setup mediapipe ----------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --------- Try obtaining system master volume ----------
volume_interface, method_used = get_master_volume_interface()

if volume_interface is None:
    print(f"INFO: master volume NOT available → Using per-app session volume (method={method_used})")
else:
    try:
        vrange = volume_interface.GetVolumeRange()
        minVol, maxVol = float(vrange[0]), float(vrange[1])
        print(f"INFO: master volume active via {method_used}  Range=({minVol}, {maxVol})")
    except:
        minVol, maxVol = FALLBACK_MIN_VOL, FALLBACK_MAX_VOL
        print("WARNING: Could not read master volume range. Using fallback values.")


# --------- Webcam setup ----------
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

# --------- Mediapipe Hands tracking ----------
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

    while True:
        success, frame = cam.read()
        if not success:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        lmList = []
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for lm in results.multi_hand_landmarks[0].landmark:
                lmList.append([int(lm.x * w), int(lm.y * h)])
            mp_drawing.draw_landmarks(
                frame,
                results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
            )

        if len(lmList) >= 9:
            x1, y1 = lmList[4]   # Thumb tip
            x2, y2 = lmList[8]   # Index tip

            cv2.circle(frame, (x1, y1), 10, (255,255,255), -1)
            cv2.circle(frame, (x2, y2), 10, (255,255,255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 3)

            length = math.hypot(x2 - x1, y2 - y1)

            # ---- map hand distance to volume ----
            vol_dB = float(np.interp(length, [50, 220], [minVol, maxVol]))
            vol_pct = int(np.interp(length, [50, 220], [0, 100]))
            vol_scalar = float(np.interp(length, [50, 220], [0.0, 1.0]))

            # ---- Try MASTER VOLUME first ----
            used_master = False
            if volume_interface is not None:
                try:
                    volume_interface.SetMasterVolumeLevel(vol_dB, None)
                    used_master = True
                except:
                    used_master = False

            # ---- If master volume fails → per-app fallback ----
            if not used_master:
                try:
                    sessions = AudioUtilities.GetAllSessions()
                    for s in sessions:
                        if hasattr(s, "SimpleAudioVolume"):
                            s.SimpleAudioVolume.SetMasterVolume(vol_scalar, None)
                except:
                    pass

            # ---- Draw Volume Meter ----
            cv2.rectangle(frame, (50,150), (85,400), (0,0,0), 3)
            bar = int(np.interp(length, [50, 220], [400, 150]))
            cv2.rectangle(frame, (50, bar), (85,400), (0,255,0), -1)
            cv2.putText(frame, f"{vol_pct}%", (40,440),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 3)

        cv2.putText(frame, f"Method: {method_used}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        cv2.imshow("Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cam.release()
cv2.destroyAllWindows()
