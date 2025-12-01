# """
# Gesture Volume + Media Controller
# - Pinch (thumb+index) on RIGHT hand -> control system/app volume (and mute when pinch small)
# - Open palm (5 fingers) -> Play/Pause
# - Two fingers (index+middle) -> Next track
# - Three fingers (index+middle+ring) -> Previous track
# - Tries to set master system volume; falls back to per-app session volumes
# """

# import time
# import math
# import cv2
# import numpy as np
# import mediapipe as mp
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL, CoInitialize
# from comtypes.client import CreateObject
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# import ctypes

# # ---------- Media key virtual codes (Windows) ----------
# VK_MEDIA_NEXT = 0xB0
# VK_MEDIA_PREV = 0xB1
# VK_MEDIA_PLAY_PAUSE = 0xB3

# # ---------- Config ----------
# CONTROL_APPS = ["spotify.exe", "chrome.exe", "vlc.exe"]
# PINCH_MUTE_THRESHOLD = 35
# PINCH_MIN = 50
# PINCH_MAX = 220
# FALLBACK_MIN_VOL = -65.25
# FALLBACK_MAX_VOL = 0.0

# # Gesture cooldowns (seconds)
# GESTURE_COOLDOWN = 1.0

# # ---------- helper: send media key ----------
# _user32 = ctypes.windll.user32
# def press_media_key(vk):
#     """Simulate a media key press on Windows."""
#     KEYEVENTF_EXTENDEDKEY = 0x0001
#     KEYEVENTF_KEYUP = 0x0002
#     _user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
#     time.sleep(0.03)
#     _user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

# # ---------- pycaw master retrieval ----------
# def get_master_volume_interface():
#     try:
#         dev = AudioUtilities.GetSpeakers()
#         if hasattr(dev, "Activate"):
#             iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#             vol = cast(iface, POINTER(IAudioEndpointVolume))
#             return vol, "pycaw:GetSpeakers()"
#     except Exception:
#         pass
#     try:
#         CoInitialize()
#         enumerator = CreateObject("MMDeviceEnumerator.MMDeviceEnumerator")
#         device = enumerator.GetDefaultAudioEndpoint(0, 1)
#         iface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#         vol = cast(iface, POINTER(IAudioEndpointVolume))
#         return vol, "comtypes:MMDeviceEnumerator"
#     except Exception:
#         pass
#     try:
#         devices = AudioUtilities.GetAllDevices()
#         for d in devices:
#             if hasattr(d, "Activate"):
#                 iface = d.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#                 vol = cast(iface, POINTER(IAudioEndpointVolume))
#                 return vol, "pycaw:GetAllDevices()"
#     except Exception:
#         pass
#     return None, "none"

# def set_all_sessions_volume_scalar(scalar):
#     try:
#         sessions = AudioUtilities.GetAllSessions()
#         for s in sessions:
#             try:
#                 if hasattr(s, "SimpleAudioVolume"):
#                     s.SimpleAudioVolume.SetMasterVolume(float(scalar), None)
#             except Exception:
#                 pass
#     except Exception:
#         pass

# def set_sessions_mute(mute_bool):
#     try:
#         sessions = AudioUtilities.GetAllSessions()
#         for s in sessions:
#             try:
#                 if hasattr(s, "SimpleAudioVolume"):
#                     s.SimpleAudioVolume.SetMute(bool(mute_bool), None)
#             except Exception:
#                 pass
#     except Exception:
#         pass

# def set_specific_apps_volume_scalar(scalar, apps):
#     try:
#         sessions = AudioUtilities.GetAllSessions()
#         for s in sessions:
#             try:
#                 proc = getattr(s, "Process", None)
#                 if proc and proc.name():
#                     name = proc.name().lower()
#                     for a in apps:
#                         if name == a.lower():
#                             if hasattr(s, "SimpleAudioVolume"):
#                                 s.SimpleAudioVolume.SetMasterVolume(float(scalar), None)
#             except Exception:
#                 pass
#     except Exception:
#         pass

# # ---------- Init volume interface ----------
# volume_interface, vol_method = get_master_volume_interface()
# if volume_interface is None:
#     print(f"INFO: master volume unavailable, falling back to per-app sessions (method={vol_method})")
#     minVol, maxVol = FALLBACK_MIN_VOL, FALLBACK_MAX_VOL
# else:
#     try:
#         vrange = volume_interface.GetVolumeRange()
#         minVol, maxVol = float(vrange[0]), float(vrange[1])
#         print(f"INFO: master volume available via {vol_method}. Range=({minVol}, {maxVol})")
#     except Exception:
#         minVol, maxVol = FALLBACK_MIN_VOL, FALLBACK_MAX_VOL
#         print("WARNING: could not read master range; using fallback values")

# # ---------- Mediapipe + camera ----------
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# mp_styles = mp.solutions.drawing_styles

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# prev_time = time.time()
# fps = 0.0

# # last gesture timestamps to avoid repeating
# last_gesture_time = 0.0
# last_gesture_name = ""

# def fingers_up_from_landmarks(lm, handedness_label):
#     """
#     lm: list of [x,y] coordinates for 21 landmarks
#     handedness_label: 'Right' or 'Left'
#     returns list [thumb, index, middle, ring, pinky] (1 if up)
#     """
#     tip_ids = [4, 8, 12, 16, 20]
#     fingers = []

#     # Thumb: compare x with IP joint (id 3)
#     try:
#         if handedness_label.lower() == 'right':
#             fingers.append(1 if lm[4][0] > lm[3][0] else 0)
#         else:
#             fingers.append(1 if lm[4][0] < lm[3][0] else 0)
#     except Exception:
#         fingers.append(0)

#     # Other fingers: tip y < pip y => finger is up (image origin top-left)
#     for id in [8, 12, 16, 20]:
#         try:
#             fingers.append(1 if lm[id][1] < lm[id - 2][1] else 0)
#         except Exception:
#             fingers.append(0)

#     return fingers

# with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             continue

#         h, w, _ = frame.shape
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)
#         frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

#         # collect hands + handedness
#         hands_data = []
#         if results.multi_hand_landmarks and results.multi_handedness:
#             for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#                 label = hand_handedness.classification[0].label  # 'Left' or 'Right'
#                 lm_coords = []
#                 for lm in hand_landmarks.landmark:
#                     lm_coords.append([int(lm.x * w), int(lm.y * h)])
#                 hands_data.append({'label': label, 'lm': lm_coords, 'raw': hand_landmarks})
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                                        mp_styles.get_default_hand_landmarks_style(),
#                                        mp_styles.get_default_hand_connections_style())

#         # FPS smoothing
#         cur_time = time.time()
#         raw_fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
#         fps = fps * 0.9 + raw_fps * 0.1
#         prev_time = cur_time

#         vol_percent = None
#         mode_label = "idle"
#         mute_state = None

#         # Choose target hand (prefer RIGHT)
#         target_hand = None
#         for hd in hands_data:
#             if hd['label'].lower() == 'right':
#                 target_hand = hd
#                 break
#         if target_hand is None and len(hands_data) > 0:
#             target_hand = hands_data[0]

#         # Process gestures for media (open palm / 2-fingers / 3-fingers)
#         gesture_triggered = None
#         if target_hand:
#             fingers = fingers_up_from_landmarks(target_hand['lm'], target_hand['label'])
#             fingers_count = sum(fingers[1:]) + fingers[0]  # quick count
#             # But we'll use specific patterns
#             # Exclude when pinch (thumb+index) is used for volume
#             x1, y1 = target_hand['lm'][4]
#             x2, y2 = target_hand['lm'][8]
#             pinch_dist = math.hypot(x2 - x1, y2 - y1)

#             is_pinch = pinch_dist < PINCH_MAX  # pinch active if within range (we'll use more exact below)
#             # If pinch (we use for volume) -> handle volume below and skip media gestures
#             if pinch_dist <= PINCH_MAX and pinch_dist >= 0:
#                 pass
#             else:
#                 # interpret gestures only when NOT pinching
#                 # pattern detection:
#                 # all five fingers up -> play/pause
#                 if fingers == [1,1,1,1,1]:
#                     gesture_triggered = "play_pause"
#                 # two fingers up (index + middle) -> next
#                 elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
#                     gesture_triggered = "next"
#                 # three fingers up (index + middle + ring) -> previous
#                 elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
#                     gesture_triggered = "previous"

#         # If a gesture is detected, ensure cooldown
#         now = time.time()
#         if gesture_triggered:
#             if now - last_gesture_time > GESTURE_COOLDOWN or gesture_triggered != last_gesture_name:
#                 # execute action
#                 if gesture_triggered == "play_pause":
#                     press_media_key(VK_MEDIA_PLAY_PAUSE)
#                     mode_label = "Play/Pause"
#                 elif gesture_triggered == "next":
#                     press_media_key(VK_MEDIA_NEXT)
#                     mode_label = "Next Track"
#                 elif gesture_triggered == "previous":
#                     press_media_key(VK_MEDIA_PREV)
#                     mode_label = "Previous Track"
#                 last_gesture_time = now
#                 last_gesture_name = gesture_triggered
#         else:
#             # no media gesture -> clear last_gesture_name after cooldown to allow repeat
#             if now - last_gesture_time > GESTURE_COOLDOWN:
#                 last_gesture_name = ""

#         # Volume control via pinch (thumb+index)
#         if target_hand:
#             x1, y1 = target_hand['lm'][4]
#             x2, y2 = target_hand['lm'][8]
#             cv2.circle(frame, (x1, y1), 10, (255,255,255), -1)
#             cv2.circle(frame, (x2, y2), 10, (255,255,255), -1)
#             cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 3)
#             length = math.hypot(x2 - x1, y2 - y1)

#             # map to dB & scalar
#             vol_db = float(np.interp(length, [PINCH_MIN, PINCH_MAX], [minVol, maxVol]))
#             vol_scalar = float(np.interp(length, [PINCH_MIN, PINCH_MAX], [0.0, 1.0]))
#             vol_percent = int(np.interp(length, [PINCH_MIN, PINCH_MAX], [0, 100]))

#             # mute detection
#             if length <= PINCH_MUTE_THRESHOLD:
#                 mute_state = True
#             else:
#                 mute_state = False

#             master_ok = False
#             if volume_interface is not None:
#                 try:
#                     volume_interface.SetMasterVolumeLevel(vol_db, None)
#                     volume_interface.SetMute(1 if mute_state else 0, None)
#                     master_ok = True
#                     mode_label = "Volume (master)"
#                 except Exception:
#                     master_ok = False

#             if not master_ok:
#                 set_all_sessions_volume_scalar(vol_scalar)
#                 set_sessions_mute(mute_state)
#                 mode_label = "Volume (sessions)"

#             # adjust specific apps too
#             set_specific_apps_volume_scalar(vol_scalar, CONTROL_APPS)

#         # Overlay
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (10, 10), (340, 140), (0,0,0), -1)
#         alpha = 0.35
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#         if vol_percent is not None:
#             cv2.putText(frame, f"Volume: {vol_percent}%", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
#         else:
#             cv2.putText(frame, "Volume: --", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180,180,180), 2)

#         if mute_state is not None:
#             cv2.putText(frame, "Muted" if mute_state else "Unmuted", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

#         cv2.putText(frame, f"Mode: {mode_label}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 1)
#         cv2.putText(frame, f"Vol method: {vol_method}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
#         cv2.putText(frame, f"FPS: {int(fps)}", (260, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
#         cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140,140,140), 1)

#         cv2.imshow("Gesture Volume + Media", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

# cap.release()
# cv2.destroyAllWindows()
# """
# Fixed Gesture Volume + Media Controller
# - Pinch (thumb+index) -> volume control (and mute when very close)
# - Open palm (all 5 fingers) -> Play/Pause
# - Two fingers (index+middle) -> Next track
# - Three fingers (index+middle+ring) -> Previous track

# Important fix: media gestures only run when NOT in the volume pinch-range.
# """

# import time
# import math
# import cv2
# import numpy as np
# import mediapipe as mp
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL, CoInitialize
# from comtypes.client import CreateObject
# from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# import ctypes

# # ---------- Media key virtual codes (Windows) ----------
# VK_MEDIA_NEXT = 0xB0
# VK_MEDIA_PREV = 0xB1
# VK_MEDIA_PLAY_PAUSE = 0xB3

# # ---------- Config ----------
# CONTROL_APPS = ["spotify.exe", "chrome.exe", "vlc.exe"]
# PINCH_MUTE_THRESHOLD = 35       # px: very close -> mute
# PINCH_MIN = 50                  # px: minimum distance mapped to min volume
# PINCH_MAX = 220                 # px: maximum distance mapped to max volume
# FALLBACK_MIN_VOL = -65.25
# FALLBACK_MAX_VOL = 0.0
# GESTURE_COOLDOWN = 1.0          # seconds between repeating same gesture

# # ---------- helper: send media key ----------
# _user32 = ctypes.windll.user32
# def press_media_key(vk):
#     """Simulate a media key press on Windows."""
#     KEYEVENTF_EXTENDEDKEY = 0x0001
#     KEYEVENTF_KEYUP = 0x0002
#     _user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
#     time.sleep(0.03)
#     _user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

# # ---------- pycaw helpers ----------
# def get_master_volume_interface():
#     try:
#         dev = AudioUtilities.GetSpeakers()
#         if hasattr(dev, "Activate"):
#             iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#             vol = cast(iface, POINTER(IAudioEndpointVolume))
#             return vol, "pycaw:GetSpeakers()"
#     except Exception:
#         pass
#     try:
#         CoInitialize()
#         enumerator = CreateObject("MMDeviceEnumerator.MMDeviceEnumerator")
#         device = enumerator.GetDefaultAudioEndpoint(0, 1)
#         iface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#         vol = cast(iface, POINTER(IAudioEndpointVolume))
#         return vol, "comtypes:MMDeviceEnumerator"
#     except Exception:
#         pass
#     try:
#         devices = AudioUtilities.GetAllDevices()
#         for d in devices:
#             if hasattr(d, "Activate"):
#                 iface = d.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
#                 vol = cast(iface, POINTER(IAudioEndpointVolume))
#                 return vol, "pycaw:GetAllDevices()"
#     except Exception:
#         pass
#     return None, "none"

# def set_all_sessions_volume_scalar(scalar):
#     try:
#         sessions = AudioUtilities.GetAllSessions()
#         for s in sessions:
#             try:
#                 if hasattr(s, "SimpleAudioVolume"):
#                     s.SimpleAudioVolume.SetMasterVolume(float(scalar), None)
#             except Exception:
#                 pass
#     except Exception:
#         pass

# def set_sessions_mute(mute_bool):
#     try:
#         sessions = AudioUtilities.GetAllSessions()
#         for s in sessions:
#             try:
#                 if hasattr(s, "SimpleAudioVolume"):
#                     s.SimpleAudioVolume.SetMute(bool(mute_bool), None)
#             except Exception:
#                 pass
#     except Exception:
#         pass

# def set_specific_apps_volume_scalar(scalar, apps):
#     try:
#         sessions = AudioUtilities.GetAllSessions()
#         for s in sessions:
#             try:
#                 proc = getattr(s, "Process", None)
#                 if proc and proc.name():
#                     name = proc.name().lower()
#                     for a in apps:
#                         if name == a.lower():
#                             if hasattr(s, "SimpleAudioVolume"):
#                                 s.SimpleAudioVolume.SetMasterVolume(float(scalar), None)
#             except Exception:
#                 pass
#     except Exception:
#         pass

# # ---------- Init volume interface ----------
# volume_interface, vol_method = get_master_volume_interface()
# if volume_interface is None:
#     print(f"INFO: master volume unavailable, falling back to per-app sessions (method={vol_method})")
#     minVol, maxVol = FALLBACK_MIN_VOL, FALLBACK_MAX_VOL
# else:
#     try:
#         vrange = volume_interface.GetVolumeRange()
#         minVol, maxVol = float(vrange[0]), float(vrange[1])
#         print(f"INFO: master volume available via {vol_method}. Range=({minVol}, {maxVol})")
#     except Exception:
#         minVol, maxVol = FALLBACK_MIN_VOL, FALLBACK_MAX_VOL
#         print("WARNING: could not read master range; using fallback values")

# # ---------- Mediapipe + camera ----------
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# mp_styles = mp.solutions.drawing_styles

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# prev_time = time.time()
# fps = 0.0
# last_gesture_time = 0.0
# last_gesture_name = ""

# def fingers_up_from_landmarks(lm, handedness_label):
#     tip_ids = [4, 8, 12, 16, 20]
#     fingers = []
#     # Thumb: consider hand orientation
#     try:
#         if handedness_label.lower() == 'right':
#             fingers.append(1 if lm[4][0] > lm[3][0] else 0)
#         else:
#             fingers.append(1 if lm[4][0] < lm[3][0] else 0)
#     except Exception:
#         fingers.append(0)
#     # Other fingers by comparing tip and pip y
#     for id in [8, 12, 16, 20]:
#         try:
#             fingers.append(1 if lm[id][1] < lm[id - 2][1] else 0)
#         except Exception:
#             fingers.append(0)
#     return fingers

# with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             continue

#         h, w, _ = frame.shape
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)
#         frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

#         # collect hands
#         hands_data = []
#         if results.multi_hand_landmarks and results.multi_handedness:
#             for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#                 label = hand_handedness.classification[0].label  # 'Left' or 'Right'
#                 lm_coords = []
#                 for lm in hand_landmarks.landmark:
#                     lm_coords.append([int(lm.x * w), int(lm.y * h)])
#                 hands_data.append({'label': label, 'lm': lm_coords})
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                                        mp_styles.get_default_hand_landmarks_style(),
#                                        mp_styles.get_default_hand_connections_style())

#         # FPS
#         cur_time = time.time()
#         raw_fps = 1.0 / (cur_time - prev_time) if (cur_time - prev_time) > 0 else 0.0
#         fps = fps * 0.9 + raw_fps * 0.1
#         prev_time = cur_time

#         vol_percent = None
#         mode_label = "idle"
#         mute_state = None

#         # select target hand (prefer right)
#         target_hand = None
#         for hd in hands_data:
#             if hd['label'].lower() == 'right':
#                 target_hand = hd
#                 break
#         if target_hand is None and len(hands_data) > 0:
#             target_hand = hands_data[0]

#         gesture_triggered = None

#         # If we have a target hand, evaluate pinch distance and finger states
#         if target_hand:
#             lm = target_hand['lm']
#             # ensure enough landmarks
#             if len(lm) >= 9:
#                 x1, y1 = lm[4]   # thumb tip
#                 x2, y2 = lm[8]   # index tip
#                 pinch_dist = math.hypot(x2 - x1, y2 - y1)

#                 # Determine whether we are in the VOLUME control range
#                 in_volume_range = (PINCH_MIN <= pinch_dist <= PINCH_MAX)

#                 # If NOT in volume range -> allow media gestures based on fingers pattern
#                 if not in_volume_range:
#                     fingers = fingers_up_from_landmarks(lm, target_hand['label'])
#                     # Determine patterns
#                     if fingers == [1,1,1,1,1]:
#                         gesture_triggered = "play_pause"
#                     elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
#                         gesture_triggered = "next"
#                     elif fingers[0] == 0 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
#                         gesture_triggered = "previous"
#                 # If in volume range -> handle volume & mute and skip media gestures
#                 else:
#                     # draw indicators for volume
#                     cv2.circle(frame, (x1, y1), 10, (255,255,255), -1)
#                     cv2.circle(frame, (x2, y2), 10, (255,255,255), -1)
#                     cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 3)

#                     length = pinch_dist
#                     vol_db = float(np.interp(length, [PINCH_MIN, PINCH_MAX], [minVol, maxVol]))
#                     vol_scalar = float(np.interp(length, [PINCH_MIN, PINCH_MAX], [0.0, 1.0]))
#                     vol_percent = int(np.interp(length, [PINCH_MIN, PINCH_MAX], [0, 100]))

#                     # mute when very close
#                     if length <= PINCH_MUTE_THRESHOLD:
#                         mute_state = True
#                     else:
#                         mute_state = False

#                     master_ok = False
#                     if volume_interface is not None:
#                         try:
#                             volume_interface.SetMasterVolumeLevel(vol_db, None)
#                             volume_interface.SetMute(1 if mute_state else 0, None)
#                             master_ok = True
#                             mode_label = "Volume (master)"
#                         except Exception:
#                             master_ok = False

#                     if not master_ok:
#                         set_all_sessions_volume_scalar(vol_scalar)
#                         set_sessions_mute(mute_state)
#                         mode_label = "Volume (sessions)"

#                     # adjust specific apps
#                     set_specific_apps_volume_scalar(vol_scalar, CONTROL_APPS)

#         # Handle a detected media gesture (with cooldown)
#         now = time.time()
#         if gesture_triggered:
#             if now - last_gesture_time > GESTURE_COOLDOWN or gesture_triggered != last_gesture_name:
#                 if gesture_triggered == "play_pause":
#                     press_media_key(VK_MEDIA_PLAY_PAUSE)
#                     mode_label = "Play/Pause"
#                 elif gesture_triggered == "next":
#                     press_media_key(VK_MEDIA_NEXT)
#                     mode_label = "Next Track"
#                 elif gesture_triggered == "previous":
#                     press_media_key(VK_MEDIA_PREV)
#                     mode_label = "Previous Track"
#                 last_gesture_time = now
#                 last_gesture_name = gesture_triggered
#         else:
#             if now - last_gesture_time > GESTURE_COOLDOWN:
#                 last_gesture_name = ""

#         # Overlay UI
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (10, 10), (340, 140), (0,0,0), -1)
#         alpha = 0.35
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#         if vol_percent is not None:
#             cv2.putText(frame, f"Volume: {vol_percent}%", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
#         else:
#             cv2.putText(frame, "Volume: --", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180,180,180), 2)

#         if mute_state is not None:
#             cv2.putText(frame, "Muted" if mute_state else "Unmuted", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

#         cv2.putText(frame, f"Mode: {mode_label}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 1)
#         cv2.putText(frame, f"Vol method: {vol_method}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
#         cv2.putText(frame, f"FPS: {int(fps)}", (260, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
#         cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140,140,140), 1)

#         cv2.imshow("Gesture Volume + Media", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

# cap.release()
# cv2.destroyAllWindows()
"""
Robust Gesture Volume + Media Controller (improved gesture detection)

- Pinch (thumb+index) -> volume control & mute when very close
- Open palm (all 5 fingers spread) -> Play/Pause
- Two fingers (index+middle) -> Next track
- Three fingers (index+middle+ring) -> Previous track

This version uses distance thresholds and spread checks to increase real-world reliability.
It prints debug lines to console when a gesture is recognized.
"""

import time
import math
import cv2
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL, CoInitialize
from comtypes.client import CreateObject
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import ctypes

# ---------- Media key virtual codes (Windows) ----------
VK_MEDIA_NEXT = 0xB0
VK_MEDIA_PREV = 0xB1
VK_MEDIA_PLAY_PAUSE = 0xB3

# ---------- Config ----------
CONTROL_APPS = ["spotify.exe", "chrome.exe", "vlc.exe"]
PINCH_MUTE_THRESHOLD = 35       # px
PINCH_MIN = 50
PINCH_MAX = 220
FALLBACK_MIN_VOL = -65.25
FALLBACK_MAX_VOL = 0.0
GESTURE_COOLDOWN = 1.0

# Debug toggles
PRINT_DEBUG = True      # prints finger flags every frame when True
PRINT_ON_FIRE = True    # prints message when gesture triggers

# ---------- helper: send media key ----------
_user32 = ctypes.windll.user32
def press_media_key(vk):
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002
    _user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
    time.sleep(0.03)
    _user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

# ---------- pycaw helpers ----------
def get_master_volume_interface():
    try:
        dev = AudioUtilities.GetSpeakers()
        if hasattr(dev, "Activate"):
            iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            vol = cast(iface, POINTER(IAudioEndpointVolume))
            return vol, "pycaw:GetSpeakers()"
    except Exception:
        pass
    try:
        CoInitialize()
        enumerator = CreateObject("MMDeviceEnumerator.MMDeviceEnumerator")
        device = enumerator.GetDefaultAudioEndpoint(0, 1)
        iface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        vol = cast(iface, POINTER(IAudioEndpointVolume))
        return vol, "comtypes:MMDeviceEnumerator"
    except Exception:
        pass
    try:
        devices = AudioUtilities.GetAllDevices()
        for d in devices:
            if hasattr(d, "Activate"):
                iface = d.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                vol = cast(iface, POINTER(IAudioEndpointVolume))
                return vol, "pycaw:GetAllDevices()"
    except Exception:
        pass
    return None, "none"

def set_all_sessions_volume_scalar(scalar):
    try:
        sessions = AudioUtilities.GetAllSessions()
        for s in sessions:
            try:
                if hasattr(s, "SimpleAudioVolume"):
                    s.SimpleAudioVolume.SetMasterVolume(float(scalar), None)
            except Exception:
                pass
    except Exception:
        pass

def set_sessions_mute(mute_bool):
    try:
        sessions = AudioUtilities.GetAllSessions()
        for s in sessions:
            try:
                if hasattr(s, "SimpleAudioVolume"):
                    s.SimpleAudioVolume.SetMute(bool(mute_bool), None)
            except Exception:
                pass
    except Exception:
        pass

def set_specific_apps_volume_scalar(scalar, apps):
    try:
        sessions = AudioUtilities.GetAllSessions()
        for s in sessions:
            try:
                proc = getattr(s, "Process", None)
                if proc and proc.name():
                    name = proc.name().lower()
                    for a in apps:
                        if name == a.lower():
                            if hasattr(s, "SimpleAudioVolume"):
                                s.SimpleAudioVolume.SetMasterVolume(float(scalar), None)
            except Exception:
                pass
    except Exception:
        pass

# ---------- Init volume interface ----------
volume_interface, vol_method = get_master_volume_interface()
if volume_interface is None:
    print(f"INFO: master volume unavailable, falling back to per-app sessions (method={vol_method})")
    minVol, maxVol = FALLBACK_MIN_VOL, FALLBACK_MAX_VOL
else:
    try:
        vrange = volume_interface.GetVolumeRange()
        minVol, maxVol = float(vrange[0]), float(vrange[1])
        print(f"INFO: master volume available via {vol_method}. Range=({minVol}, {maxVol})")
    except Exception:
        minVol, maxVol = FALLBACK_MIN_VOL, FALLBACK_MAX_VOL
        print("WARNING: could not read master range; using fallback values")

# ---------- Mediapipe + camera ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# gesture cooldown
last_trigger_time = {"play_pause": 0.0, "next": 0.0, "previous": 0.0}

print("Starting. Make sure an audio app is playing to hear volume changes and media keys to be effective.")

def finger_flags_from_landmarks(lm, handedness_label):
    """
    Returns a dict with boolean flags for thumb, index, middle, ring, pinky.
    Uses tip vs pip distances AND tip-to-wrist distance thresholds for robustness.
    lm = list of 21 [x,y] points (pixel coords)
    """
    flags = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
    try:
        wrist = lm[0]
        # distances helper
        def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

        # Thumb: check tip vs ip (4 vs 3) in x-direction for orientation + some distance from wrist
        try:
            tip, ip = lm[4], lm[3]
            palm = lm[0]
            # threshold_px = relative to hand width; compute hand width approx by distance between index_mcp(5) and pinky_mcp(17)
            hand_width = dist(lm[5], lm[17]) if (lm[5] and lm[17]) else 60
            xgap = tip[0] - ip[0]
            if handedness_label.lower() == 'right':
                flags["thumb"] = xgap > (hand_width * 0.08) and dist(tip, palm) > (hand_width * 0.12)
            else:
                flags["thumb"] = xgap < -(hand_width * 0.08) and dist(tip, palm) > (hand_width * 0.12)
        except Exception:
            flags["thumb"] = False

        # Other fingers: tip id vs pip id (tip id < pip id y) AND tip distance from wrist sufficiently large
        for name, tip_id, pip_id in [("index",8,6), ("middle",12,10), ("ring",16,14), ("pinky",20,18)]:
            try:
                tip = lm[tip_id]
                pip = lm[pip_id]
                tip_wrist_dist = dist(tip, wrist)
                # dynamic thresholds relative to hand size
                hand_size = dist(lm[5], lm[17]) if (lm[5] and lm[17]) else 80
                # finger is up if tip is higher than pip (smaller y) by margin and tip is some distance from wrist
                flags[name] = (tip[1] < pip[1] - (hand_size * 0.03)) and (tip_wrist_dist > (hand_size * 0.18))
            except Exception:
                flags[name] = False
    except Exception:
        pass
    return flags

def is_open_palm(flags, lm):
    """Open palm: all fingers True AND spread wide (max_x - min_x) larger than threshold"""
    if not all(flags.values()):
        return False
    xs = [p[0] for p in lm]
    spread = max(xs) - min(xs)
    # require hand spread relative to width
    return spread > 160 or spread > ( (max(xs)-min(xs)) * 0.9 )  # fallback

def is_two_finger_v(flags):
    # index & middle True; others False OR thumb allowed False/True
    return flags["index"] and flags["middle"] and (not flags["ring"]) and (not flags["pinky"])

def is_three_finger(flags):
    return flags["index"] and flags["middle"] and flags["ring"] and (not flags["pinky"])

# main loop
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # collect hands with handedness
    hands_data = []
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Left' or 'Right'
            lm_coords = []
            for lm in hand_landmarks.landmark:
                lm_coords.append([int(lm.x * w), int(lm.y * h)])
            hands_data.append({'label': label, 'lm': lm_coords})
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_styles.get_default_hand_landmarks_style(),
                                   mp_styles.get_default_hand_connections_style())

    # choose target hand (prefer right)
    target = None
    for hd in hands_data:
        if hd['label'].lower() == 'right':
            target = hd
            break
    if target is None and hands_data:
        target = hands_data[0]

    # default overlays
    vol_percent = None
    mute_state = None
    mode_label = "idle"
    gesture_to_fire = None

    if target and len(target['lm']) >= 9:
        lm = target['lm']
        flags = finger_flags_from_landmarks(lm, target['label'])

        # show flags on-screen
        flag_text = f"T{int(flags['thumb'])} I{int(flags['index'])} M{int(flags['middle'])} R{int(flags['ring'])} P{int(flags['pinky'])}"
        cv2.putText(frame, flag_text, (10, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # compute pinch distance
        x1, y1 = lm[4]
        x2, y2 = lm[8]
        pinch_dist = math.hypot(x2 - x1, y2 - y1)
        cv2.putText(frame, f"pinch:{int(pinch_dist)}", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        # Determine if in volume range (pinch between PINCH_MIN and PINCH_MAX)
        in_volume_range = (PINCH_MIN <= pinch_dist <= PINCH_MAX)

        if in_volume_range:
            # Volume control
            length = pinch_dist
            vol_db = float(np.interp(length, [PINCH_MIN, PINCH_MAX], [minVol, maxVol]))
            vol_scalar = float(np.interp(length, [PINCH_MIN, PINCH_MAX], [0.0, 1.0]))
            vol_percent = int(np.interp(length, [PINCH_MIN, PINCH_MAX], [0, 100]))
            mute_state = length <= PINCH_MUTE_THRESHOLD

            # attempt master
            master_ok = False
            if volume_interface is not None:
                try:
                    volume_interface.SetMasterVolumeLevel(vol_db, None)
                    volume_interface.SetMute(1 if mute_state else 0, None)
                    master_ok = True
                    mode_label = "volume(master)"
                except Exception:
                    master_ok = False
            if not master_ok:
                set_all_sessions_volume_scalar(vol_scalar)
                set_sessions_mute(mute_state)
                mode_label = "volume(sessions)"
            # specific apps
            set_specific_apps_volume_scalar(vol_scalar, CONTROL_APPS)

        else:
            # Not in volume mode -> check media gestures with relaxed rules
            # Open palm detection
            if is_open_palm(flags, lm):
                gesture_to_fire = "play_pause"
            # Two finger V (index+middle)
            elif is_two_finger_v(flags):
                gesture_to_fire = "next"
            # Three fingers (index+middle+ring)
            elif is_three_finger(flags):
                gesture_to_fire = "previous"

            if PRINT_DEBUG:
                # print debug flags to console occasionally
                print(f"[DEBUG] flags={flags} pinch={int(pinch_dist)} -> gesture_candidate={gesture_to_fire}")

    # Fire gesture if any, with cooldown
    now = time.time()
    if gesture_to_fire:
        last = last_trigger_time.get(gesture_to_fire, 0.0)
        if now - last > GESTURE_COOLDOWN:
            if PRINT_ON_FIRE:
                print(f"[FIRE] {gesture_to_fire} @ {time.strftime('%H:%M:%S')}")
            if gesture_to_fire == "play_pause":
                press_media_key(VK_MEDIA_PLAY_PAUSE)
                mode_label = "Play/Pause"
            elif gesture_to_fire == "next":
                press_media_key(VK_MEDIA_NEXT)
                mode_label = "Next"
            elif gesture_to_fire == "previous":
                press_media_key(VK_MEDIA_PREV)
                mode_label = "Previous"
            last_trigger_time[gesture_to_fire] = now

    # Draw overlay panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (380,140), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    if vol_percent is not None:
        cv2.putText(frame, f"Volume: {vol_percent}%", (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    else:
        cv2.putText(frame, "Volume: --", (20,45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180,180,180), 2)

    if mute_state is not None:
        cv2.putText(frame, "Muted" if mute_state else "Unmuted", (200,45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

    cv2.putText(frame, f"Mode: {mode_label}", (20,85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 1)
    cv2.putText(frame, f"Vol method: {vol_method}", (20,110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    cv2.imshow("Gesture Volume + Media (robust)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
hands_detector.close()
cap.release()
cv2.destroyAllWindows()
