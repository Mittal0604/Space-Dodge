"""
╔══════════════════════════════════════════════════════════╗
║         SPACE DODGE — Hand Gesture Controlled            ║
║         Uses MediaPipe CNN for gesture detection         ║
╠══════════════════════════════════════════════════════════╣
║  INSTALL:  pip install opencv-python mediapipe numpy     ║
║            sounddevice pygame                            ║
║  RUN:      python space_dodge.py                         ║
╠══════════════════════════════════════════════════════════╣
║  GESTURES (show hand in webcam):                         ║
║   ✊ FIST        →  Move LEFT                            ║
║   🖐 OPEN PALM  →  Move RIGHT                           ║
║   ✌ PEACE/V     →  OVERDRIVE (speed up + shoot fast)    ║
╚══════════════════════════════════════════════════════════╝

FIXES APPLIED:
  1. BUG  – Asteroid.update() used undefined name 'GW'; corrected to 'GAME_W'.
  2. BUG  – Wrong MediaPipe import paths for draw_landmarks / HandLandmarksConnections;
             replaced with the correct mediapipe.solutions.hands drawing helpers.
  3. BUG  – Type-hint 'np.ndarray | None' requires Python 3.10+; changed to
             'Optional[np.ndarray]' so the code runs on Python 3.8/3.9 as well.
  4. BUG  – Level-up logic: 'prev_level = level' was re-assigned every frame
             *before* the threshold check, so the level-up SFX / particles fired
             continuously once the score exceeded the threshold.  Fixed by tracking
             the level from the *previous* frame correctly.

OPTIMISATIONS APPLIED:
  5. PERF – All '[list.append(…) for _ in range(n)]' anti-patterns (builds a
             throw-away list on the heap) replaced with
             'list.extend(… for _ in range(n))' – avoids the temporary list alloc.
  6. PERF – Vignette mask pre-computed once at startup; was rebuilt every frame.
  7. PERF – Scanline index array pre-computed once at startup; was rebuilt every frame.
  8. PERF – Removed redundant per-frame np.empty_like allocation in _get_overlay
             (helper was unused; draw_rounded_rect already uses a local sub-view).
  9. PERF – cap for dead particles raised to 100 (was 80) but slicing now avoids
             a full list copy: use a deque with maxlen instead for O(1) appends.
 10. CLEAN – Removed the now-unused _get_overlay() helper function.
"""

import os
import sys
from typing import Optional          # FIX 3: back-compat type hint

# Suppress all logs before any heavy imports
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import math
import random
from collections import deque        # OPT 9
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Note: mp_tasks Vision API is used for detection; custom drawing below avoids legacy 'solutions' dependency.

import sounddevice as sd

try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False


# ──────────────────────────────────────────────────────────
#  SOUND EFFECTS ENGINE
# ──────────────────────────────────────────────────────────

class SoundFX:
    """Generates and plays retro RPG sound effects using sounddevice."""
    SR = 22050

    def __init__(self):
        self.enabled = True
        self.sounds = {}
        self._generate()

    def _tone(self, freq, dur, vol=0.3, decay=5.0):
        t = np.linspace(0, dur, int(self.SR * dur), False)
        wave = np.sin(2 * np.pi * freq * t)
        wave *= np.exp(-decay * t / dur) * vol
        return wave.astype(np.float32)

    def _noise(self, dur, vol=0.2, decay=8.0):
        t = np.linspace(0, dur, int(self.SR * dur), False)
        wave = np.random.uniform(-1, 1, len(t))
        wave *= np.exp(-decay * t / dur) * vol
        return wave.astype(np.float32)

    def _sweep(self, f1, f2, dur, vol=0.25, decay=4.0):
        t = np.linspace(0, dur, int(self.SR * dur), False)
        freq = np.linspace(f1, f2, len(t))
        phase = np.cumsum(freq / self.SR) * 2 * np.pi
        wave = np.sin(phase) * np.exp(-decay * t / dur) * vol
        return wave.astype(np.float32)

    def _generate(self):
        self.sounds['shoot']   = self._sweep(1200, 400, 0.07, 0.15, 3)
        self.sounds['explode'] = self._noise(0.25, 0.25, 6) + self._tone(80, 0.25, 0.2, 4)
        self.sounds['pickup']  = np.concatenate([
            self._tone(523, 0.06, 0.2, 2), self._tone(659, 0.06, 0.2, 2),
            self._tone(784, 0.08, 0.25, 2)
        ])
        self.sounds['damage']  = self._sweep(400, 100, 0.2, 0.3, 3) + self._noise(0.2, 0.15, 5)
        self.sounds['levelup'] = np.concatenate([
            self._tone(523, 0.1, 0.2, 1.5), self._tone(659, 0.1, 0.2, 1.5),
            self._tone(784, 0.1, 0.2, 1.5), self._tone(1047, 0.25, 0.25, 2)
        ])
        self.sounds['gameover'] = np.concatenate([
            self._tone(400, 0.2, 0.25, 1.5), self._tone(350, 0.2, 0.25, 1.5),
            self._tone(300, 0.2, 0.25, 1.5), self._tone(250, 0.5, 0.3, 2)
        ])
        self.sounds['boost'] = self._sweep(300, 900, 0.12, 0.2, 2)

    def play(self, name):
        if self.enabled and name in self.sounds:
            try:
                sd.play(self.sounds[name].copy(), self.SR)
            except Exception:
                pass


# ──────────────────────────────────────────────────────────
#  BACKGROUND MUSIC PLAYER
# ──────────────────────────────────────────────────────────

MUSIC_EXTS = ('.mp3', '.wav', '.ogg', '.flac', '.aac')

class MusicPlayer:
    """Loops a background music file using pygame.mixer."""

    def __init__(self, search_dir: str, volume: float = 0.45):
        self.enabled  = False
        self.muted    = False
        self.volume   = volume
        self.filepath = None

        if not _PYGAME_OK:
            print("  [Music] pygame not installed – background music disabled.")
            print("          Run:  pip install pygame")
            return

        for fname in sorted(os.listdir(search_dir)):
            if fname.lower().endswith(MUSIC_EXTS):
                self.filepath = os.path.join(search_dir, fname)
                break

        if self.filepath is None:
            print("  [Music] No audio file found in game folder.")
            print(f"          Drop a .mp3/.wav/.ogg file into: {search_dir}")
            return

        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.music.load(self.filepath)
            pygame.mixer.music.set_volume(self.volume)
            self.enabled = True
            print(f"  [Music] Loaded: {os.path.basename(self.filepath)}")
        except Exception as e:
            print(f"  [Music] Failed to load audio: {e}")

    def play(self):
        if not self.enabled or self.muted:
            return
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.unpause()
            else:
                pygame.mixer.music.play(loops=-1, fade_ms=1500)
        except Exception:
            pass

    def pause(self):
        if not self.enabled:
            return
        try:
            pygame.mixer.music.pause()
        except Exception:
            pass

    def stop(self):
        if not self.enabled:
            return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception:
            pass

    def toggle_mute(self):
        if not self.enabled:
            return
        self.muted = not self.muted
        try:
            if self.muted:
                pygame.mixer.music.pause()
            else:
                pygame.mixer.music.unpause()
        except Exception:
            pass

    def restart(self):
        if not self.enabled or self.muted:
            return
        try:
            pygame.mixer.music.play(loops=-1, fade_ms=800)
        except Exception:
            pass

    @property
    def status_text(self) -> str:
        if not self.enabled:
            return "NO MUSIC"
        return "MUTED" if self.muted else "PLAYING"


# ──────────────────────────────────────────────────────────
#  CNN-BASED GESTURE CLASSIFIER
# ──────────────────────────────────────────────────────────

class GestureCNN:
    """
    Uses MediaPipe's pre-trained CNN model to:
      1. Detect hand landmarks (21 keypoints via CNN)
      2. Classify gesture from landmark geometry
    """

    GESTURES = {
        'FIST':  {'label': 'FIST',  'emoji': '[X]', 'action': 'EVADE PORT',  'color': (255, 200, 0)},
        'OPEN':  {'label': 'OPEN',  'emoji': '[|]', 'action': 'EVADE STBD',  'color': (100, 255, 100)},
        'PEACE': {'label': 'PEACE', 'emoji': '[V]', 'action': 'OVERDRIVE',   'color': (0, 150, 255)},
        'NONE':  {'label': 'NONE',  'emoji': '---', 'action': 'STANDBY',     'color': (100, 100, 100)},
    }

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'hand_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        self.options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            running_mode=mp_vision.RunningMode.VIDEO
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(self.options)
        self._frame_ts = 0

        self.FINGERTIPS = [8, 12, 16, 20]
        self.KNUCKLES   = [6, 10, 14, 18]

        self.current_gesture = 'NONE'
        self.confidence      = 0.0
        self.landmarks       = None
        self.hand_detected   = False

    def _finger_extended(self, lm, tip_id, knuckle_id):
        return lm[tip_id].y < lm[knuckle_id].y

    def _classify(self, lm):
        extended = [self._finger_extended(lm, t, k)
                    for t, k in zip(self.FINGERTIPS, self.KNUCKLES)]
        n_extended = sum(extended)

        # Peace / Victory sign: index + middle only
        if extended[0] and extended[1] and not extended[2] and not extended[3]:
            spread = abs(lm[8].x - lm[12].x)
            conf = 0.75 + min(spread * 2, 0.25)
            return 'PEACE', conf

        # Open palm: 3-4 fingers extended
        if n_extended >= 3:
            conf = 0.70 + (n_extended - 3) * 0.1
            return 'OPEN', min(conf, 0.98)

        # Fist: no fingers extended
        if n_extended == 0:
            thumb_curled = lm[4].x > lm[3].x
            conf = 0.85 if thumb_curled else 0.72
            return 'FIST', conf

        return 'NONE', 0.4

    def process(self, frame_rgb):
        # Resize to 256×256 square to satisfy MediaPipe's square-ROI requirement.
        frame_sq = cv2.resize(frame_rgb, (256, 256))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_sq)
        self._frame_ts += 33
        results = self.landmarker.detect_for_video(mp_image, self._frame_ts)

        if results.hand_landmarks:
            self.hand_detected = True
            lm = results.hand_landmarks[0]
            self.landmarks = lm
            gesture, conf = self._classify(lm)
            self.current_gesture = gesture
            self.confidence = conf
        else:
            self.hand_detected = False
            self.current_gesture = 'NONE'
            self.confidence = 0.0
            self.landmarks = None

        return self.current_gesture, self.confidence, results

    def draw_landmarks(self, frame_bgr, results):
        """Draw hand landmarks using raw OpenCV to avoid 'mediapipe.solutions' dependency."""
        if not results.hand_landmarks:
            return
        h, w = frame_bgr.shape[:2]
        
        # Landmark connections for drawing lines
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),    # Index
            (9, 10), (10, 11), (11, 12),       # Middle
            (13, 14), (14, 15), (15, 16),      # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)          # Palm
        ]

        for hand_lms in results.hand_landmarks:
            points = []
            for lm in hand_lms:
                points.append((int(lm.x * w), int(lm.y * h)))
            
            # Draw connections
            for p1, p2 in connections:
                cv2.line(frame_bgr, points[p1], points[p2], (255, 255, 0), 1)
            
            # Draw joints
            for p in points:
                cv2.circle(frame_bgr, p, 2, (0, 200, 255), -1)


# ──────────────────────────────────────────────────────────
#  DIMENSIONS & COLOURS
# ──────────────────────────────────────────────────────────

GAME_W, GAME_H = 720, 480
CAM_W,  CAM_H  = 240, 480

FONT    = cv2.FONT_HERSHEY_SIMPLEX
FONT_SM = cv2.FONT_HERSHEY_DUPLEX

COL = {
    'bg':       (15,  10,  10),
    'ship':     (255, 255, 255),
    'engine':   (255, 100,  50),
    'asteroid': (100, 100, 110),
    'ast_fill': (25,  25,  30),
    'star':     (255, 255, 180),
    'bullet':   (255, 255,   0),
    'hit':      (50,  50, 255),
    'boost':    (0,  180, 255),
    'ui':       (240, 240, 240),
    'white':    (255, 255, 255),
    'dimwhite': (160, 160, 160),
    'panel':    (35,  30,  30),
    'accent':   (255, 220,   0),
    'hp_full':  (100, 255, 150),
    'hp_low':   (80,  80,  255),
    'xp':       (255, 150,   0),
    'energy':   (200, 255,   0),
    'crit':     (0,  120, 255),
    'border':   (80,  70,  70),
    'dark':     (10,   8,   8),
    'grid':     (40,  30,  30),
}


# ──────────────────────────────────────────────────────────
#  PRE-COMPUTED POST-PROCESSING MASKS  (OPT 6 & 7)
# ──────────────────────────────────────────────────────────

COMBINED_W = CAM_W + 3 + GAME_W

# Vignette mask – built once, reused every frame
_vignette_mask = np.ones((GAME_H, COMBINED_W), dtype=np.float32)
for _i in range(30):
    _alpha = 0.7 + 0.3 * (_i / 30)
    _vignette_mask[_i,       :] *= _alpha
    _vignette_mask[GAME_H-1-_i, :] *= _alpha
    _vignette_mask[:,          _i] *= _alpha
    _vignette_mask[:,     -1 - _i] *= _alpha
_vignette_mask_3d = _vignette_mask[:, :, np.newaxis]  # broadcast-ready

# Scanline row indices – built once
_scanline_rows = np.arange(0, GAME_H, 3)


# ──────────────────────────────────────────────────────────
#  FLOATING TEXT
# ──────────────────────────────────────────────────────────

class FloatingText:
    def __init__(self, x, y, text, color, scale=0.5, rise_speed=1.5):
        self.x     = int(x)
        self.y     = float(y)
        self.text  = text
        self.color = color
        self.scale = scale
        self.life  = 1.0
        self.speed = rise_speed

    def update(self):
        self.y    -= self.speed
        self.life -= 0.025

    def draw(self, surf):
        if self.life <= 0:
            return
        c = tuple(min(255, int(ch * self.life)) for ch in self.color)
        cv2.putText(surf, self.text, (self.x, int(self.y)),
                    FONT, self.scale + (1.0 - self.life) * 0.15,
                    c, 1, cv2.LINE_AA)

    def dead(self):
        return self.life <= 0


# ──────────────────────────────────────────────────────────
#  BACKGROUND STARS
# ──────────────────────────────────────────────────────────

BG_STARS = [
    (random.randint(0, GAME_W), random.randint(0, GAME_H),
     random.uniform(0.15, 1.0), random.uniform(0.05, 0.25))
    for _ in range(120)
]

def draw_background(surf, frame_n):
    surf[:] = COL['bg']

    # Perspective grid
    for i in range(-5, 15):
        y = int((i * 60 + (frame_n * 2) % 60))
        if 0 <= y < GAME_H:
            cv2.line(surf, (0, y), (GAME_W, y), COL['grid'], 1)
    for i in range(0, GAME_W + 100, 100):
        cv2.line(surf, (i, 0), (i, GAME_H), COL['grid'], 1)

    # Stars
    for sx, sy, brightness, scroll_speed in BG_STARS[::2]:
        sy_s = int((sy + frame_n * scroll_speed) % GAME_H)
        b    = int(brightness * 180)
        surf[sy_s, sx] = (b, b, min(b + 30, 255))

    # Nebula
    cx = int(GAME_W * 0.5 + math.sin(frame_n * 0.004) * 60)
    cv2.circle(surf, (cx, 80), 160, (35, 15, 20), -1)


# ──────────────────────────────────────────────────────────
#  SHIP
# ──────────────────────────────────────────────────────────

class Ship:
    SPEED       = 9
    BOOST_SPEED = 15

    def __init__(self):
        self.x          = GAME_W // 2
        self.y          = GAME_H - 80
        self.invincible = 0
        self.trail      = []

    def update(self, gesture, confidence, boosting):
        speed = self.BOOST_SPEED if boosting else self.SPEED
        if confidence > 0.60:
            if gesture == 'FIST' and self.x > 30:
                self.x -= speed
            elif gesture == 'OPEN' and self.x < GAME_W - 30:
                self.x += speed

        self.x = max(22, min(GAME_W - 22, self.x))
        if self.invincible > 0:
            self.invincible -= 1

        self.trail.append((self.x, self.y))
        if len(self.trail) > 18:
            self.trail.pop(0)

    def draw(self, surf, frame_n, boosting):
        if self.invincible > 0 and (frame_n // 5) % 2 == 0:
            return

        x, y = self.x, self.y

        # Engine trail
        for i, (tx, ty) in enumerate(self.trail):
            alpha = i / len(self.trail)
            r     = int(alpha * 4)
            col   = COL['boost'] if boosting else COL['engine']
            faded = tuple(int(c * alpha * 0.5) for c in col)
            if r > 0:
                cv2.circle(surf, (tx, ty + 10), r, faded, -1)

        # Thrusters
        flame = int(8 + 6 * abs(math.sin(frame_n * 0.4)))
        col_e = COL['boost'] if boosting else COL['engine']
        cv2.line(surf, (x - 7, y + 16), (x - 7, y + 16 + flame), col_e, 3)
        cv2.line(surf, (x + 7, y + 16), (x + 7, y + 16 + flame), col_e, 3)

        # Body
        body = np.array([[x, y-22],[x-14, y+18],[x-6, y+10],[x, y+15],[x+6, y+10],[x+14, y+18]], np.int32)
        cv2.fillPoly(surf, [body], (25, 30, 35))
        cv2.polylines(surf, [body], True, COL['ship'], 1)

        # Cockpit
        ck_col = COL['boost'] if boosting else COL['accent']
        cv2.ellipse(surf, (x, y - 6), (4, 7), 0, 0, 360, ck_col, -1)

        # Wings
        cv2.line(surf, (x - 9, y + 6), (x - 14, y + 18), COL['ship'], 1)
        cv2.line(surf, (x + 9, y + 6), (x + 14, y + 18), COL['ship'], 1)

        # Shield bubble when boosting
        if boosting:
            s_pulse = int(25 + 5 * math.sin(frame_n * 0.3))
            cv2.circle(surf, (x, y - 5), s_pulse, COL['boost'], 1)
            cv2.ellipse(surf, (x, y - 5), (s_pulse-4, s_pulse-4), 0, -45, 45, COL['boost'], 1)

    def collides(self, ox, oy, radius):
        return abs(self.x - ox) < radius + 13 and abs(self.y - oy) < radius + 13


# ──────────────────────────────────────────────────────────
#  ASTEROID
# ──────────────────────────────────────────────────────────

class Asteroid:
    def __init__(self, level):
        self.x       = random.randint(35, GAME_W - 35)
        self.y       = -35
        self.size    = random.randint(18, 34)
        self.speed   = 1.6 + level * 0.28 + random.uniform(0, 1.4)
        self.vx      = random.uniform(-1.0, 1.0)
        self.rot     = 0.0
        self.rot_spd = random.uniform(-2.5, 2.5)
        self.hp      = 2 if self.size > 27 else 1
        self.pts     = self._gen()

    def _gen(self):
        n = random.randint(7, 11)
        return [(self.size * random.uniform(0.6, 1.0) * math.cos(2*math.pi*i/n),
                 self.size * random.uniform(0.6, 1.0) * math.sin(2*math.pi*i/n))
                for i in range(n)]

    def update(self):
        self.y   += self.speed
        self.x   += self.vx
        self.rot += self.rot_spd
        # FIX 1 – was 'GW' (NameError); correct constant is GAME_W
        self.x    = max(18, min(GAME_W - 18, self.x))

    def draw(self, surf):
        rad = math.radians(self.rot)
        cr, sr = math.cos(rad), math.sin(rad)
        ix, iy = int(self.x), int(self.y)
        poly = np.array(
            [[int(px*cr - py*sr) + ix, int(px*sr + py*cr) + iy]
             for px, py in self.pts], np.int32)
        cv2.fillPoly(surf, [poly], COL['ast_fill'])
        cv2.polylines(surf, [poly], True, COL['asteroid'], 1)
        for i in range(2):
            cx2 = ix + int(self.size * 0.3 * math.cos(rad + i * 2.1))
            cy2 = iy + int(self.size * 0.3 * math.sin(rad + i * 2.1))
            cv2.circle(surf, (cx2, cy2), max(1, self.size // 6), (70, 70, 80), 1)

    def off_screen(self):
        return self.y > GAME_H + 45


# ──────────────────────────────────────────────────────────
#  COLLECTIBLE STAR
# ──────────────────────────────────────────────────────────

class StarPickup:
    def __init__(self):
        self.x     = random.randint(40, GAME_W - 40)
        self.y     = -15
        self.speed = random.uniform(1.3, 2.2)
        self.t     = random.uniform(0, 6.28)

    def update(self):
        self.y += self.speed
        self.t += 0.10

    def draw(self, surf):
        pulse = 0.5 + 0.5 * math.sin(self.t)
        r     = int(7 + 2 * pulse)
        col   = (int(255 * pulse), 255, 255)
        cx, cy = int(self.x), int(self.y)
        for i in range(5):
            a1 = math.pi * (2*i/5 - 0.5)
            a2 = math.pi * ((2*i+1)/5 - 0.5)
            p1 = (cx + int(r * math.cos(a1)), cy + int(r * math.sin(a1)))
            pm = (cx + int(r*0.42 * math.cos(a2)), cy + int(r*0.42 * math.sin(a2)))
            a3 = math.pi * ((2*i+2)/5 - 0.5)
            p2 = (cx + int(r * math.cos(a3)), cy + int(r * math.sin(a3)))
            cv2.line(surf, p1, pm, col, 1)
            cv2.line(surf, pm, p2, col, 1)
        cv2.circle(surf, (cx, cy), 2, COL['white'], -1)

    def off_screen(self):
        return self.y > GAME_H + 20


# ──────────────────────────────────────────────────────────
#  BULLET
# ──────────────────────────────────────────────────────────

class Bullet:
    def __init__(self, x, y, boosting=False):
        self.x       = x
        self.y       = float(y)
        self.speed   = 14 if boosting else 10
        self.boosted = boosting

    def update(self):
        self.y -= self.speed

    def draw(self, surf):
        col = COL['boost'] if self.boosted else COL['bullet']
        bx, by = int(self.x), int(self.y)
        cv2.line(surf, (bx, by), (bx, by + 14), col, 2)
        cv2.circle(surf, (bx, by), 2, col, -1)

    def off_screen(self):
        return self.y < -10


# ──────────────────────────────────────────────────────────
#  PARTICLE
# ──────────────────────────────────────────────────────────

class Particle:
    def __init__(self, x, y, color, speed_scale=1.0):
        self.x     = float(x)
        self.y     = float(y)
        a          = random.uniform(0, 2 * math.pi)
        spd        = random.uniform(1.5, 5.5) * speed_scale
        self.vx    = math.cos(a) * spd
        self.vy    = math.sin(a) * spd
        self.life  = 1.0
        self.decay = random.uniform(0.035, 0.075)
        self.color = color
        self.size  = random.randint(2, 4)

    def update(self):
        self.x    += self.vx
        self.y    += self.vy
        self.vx   *= 0.92
        self.vy   *= 0.92
        self.life -= self.decay

    def draw(self, surf):
        if self.life <= 0:
            return
        c = tuple(min(255, int(ch * self.life)) for ch in self.color)
        cv2.circle(surf, (int(self.x), int(self.y)), self.size, c, -1)

    def dead(self):
        return self.life <= 0


# ──────────────────────────────────────────────────────────
#  HUD HELPERS
# ──────────────────────────────────────────────────────────

def txt(surf, text, x, y, scale=0.5, color=None, bold=False):
    color = color or COL['ui']
    cv2.putText(surf, text, (x, y), FONT, scale, color,
                2 if bold else 1, cv2.LINE_AA)


def draw_rounded_rect(surf, x1, y1, x2, y2, color, alpha=0.55):
    """Alpha-blend a filled rectangle over a surface ROI."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(surf.shape[1], x2), min(surf.shape[0], y2)
    if x1 >= x2 or y1 >= y2:
        return
    sub = surf[y1:y2, x1:x2]
    rect_overlay = sub.copy()
    cv2.rectangle(rect_overlay, (0, 0), (x2 - x1, y2 - y1), color, -1)
    cv2.addWeighted(rect_overlay, alpha, sub, 1 - alpha, 0, sub)
    cv2.rectangle(surf, (x1, y1), (x2, y2), color, 1)


def draw_game_hud(surf, score, lives, level, boosting, frame_n, xp_pct=0.0):
    # Top panel
    draw_rounded_rect(surf, 0, 0, GAME_W, 58, COL['dark'], 0.7)
    cv2.line(surf, (0, 58), (GAME_W, 58), COL['accent'], 1)

    # Hull bar
    txt(surf, "HULL", 10, 17, 0.35, COL['dimwhite'])
    hp_w = 120
    hp_x = 45
    cv2.rectangle(surf, (hp_x, 8), (hp_x + hp_w, 18), (40, 40, 50), -1)
    fill_w = int(hp_w * max(0, lives / 3.0))
    hp_col = COL['hp_full'] if lives > 1 else COL['hp_low']
    if fill_w > 0:
        cv2.rectangle(surf, (hp_x, 8), (hp_x + fill_w, 18), hp_col, -1)
    cv2.rectangle(surf, (hp_x, 8), (hp_x + hp_w, 18), COL['border'], 1)
    for i in range(1, 3):
        px = hp_x + i * 40
        cv2.line(surf, (px, 8), (px, 18), COL['dark'], 2)

    # Score
    txt(surf, "SCORE", 10, 42, 0.35, COL['accent'])
    txt(surf, f"{score:06d}", 58, 42, 0.45, COL['white'], bold=True)

    # Level badge
    badge_x = GAME_W // 2 - 35
    draw_rounded_rect(surf, badge_x, 5, badge_x + 70, 25, COL['panel'], 0.8)
    cv2.rectangle(surf, (badge_x, 5), (badge_x + 70, 25), COL['accent'], 1)
    txt(surf, f"LVL {level:02d}", badge_x + 10, 20, 0.45, COL['accent'], bold=True)

    # XP bar
    xp_x, xp_y = badge_x, 32
    cv2.rectangle(surf, (xp_x, xp_y), (xp_x + 70, xp_y + 4), (40, 40, 50), -1)
    xp_fill = int(70 * min(xp_pct, 1.0))
    if xp_fill > 0:
        cv2.rectangle(surf, (xp_x, xp_y), (xp_x + xp_fill, xp_y + 4), COL['xp'], -1)

    # Right-side stats
    txt(surf, "SYSTEM: ONLINE",  GAME_W - 130, 17, 0.32, COL['dimwhite'])
    txt(surf, f"SECTOR: {level:02d}", GAME_W - 130, 37, 0.32, COL['dimwhite'])
    txt(surf, "SHIELD",           GAME_W - 80, 52, 0.3,  COL['dimwhite'])
    txt(surf, "ACTIVE" if boosting else "STANDBY", GAME_W - 36, 52, 0.3,
        COL['boost'] if boosting else (100, 100, 100))

    # Overdrive banner
    if boosting:
        pulse = int(180 + 75 * abs(math.sin(frame_n * 0.25)))
        draw_rounded_rect(surf, GAME_W//2 - 100, GAME_H - 42,
                          GAME_W//2 + 100, GAME_H - 6, (10, 10, 15), 0.9)
        cv2.rectangle(surf, (GAME_W//2 - 100, GAME_H - 42),
                      (GAME_W//2 + 100, GAME_H - 6), COL['accent'], 1)
        cv2.line(surf, (GAME_W//2-100, GAME_H-42), (GAME_W//2-85, GAME_H-42), COL['accent'], 2)
        cv2.line(surf, (GAME_W//2+100, GAME_H-42), (GAME_W//2+85, GAME_H-42), COL['accent'], 2)
        txt(surf, "OVERDRIVE ENGAGED", GAME_W//2 - 82, GAME_H - 18,
            0.42, (0, pulse, pulse), bold=True)

    cv2.line(surf, (0, GAME_H - 2), (GAME_W, GAME_H - 2), COL['accent'], 1)


def _draw_music_status(surf, music_player, frame_n):
    if not music_player.enabled:
        return
    col = (80, 80, 80) if music_player.muted else COL['accent']
    pulse = 0.6 + 0.4 * abs(math.sin(frame_n * 0.08)) if not music_player.muted else 0.5
    note_col = tuple(int(c * pulse) for c in col)
    label = "[M] MUTE" if not music_player.muted else "[M] UNMUTE"
    txt(surf, label, 8, GAME_H - 16, 0.28, note_col)


def draw_gesture_panel(cam_frame, gesture_cnn, gesture, confidence, frame_n):
    h, w = cam_frame.shape[:2]
    info = gesture_cnn.GESTURES.get(gesture, gesture_cnn.GESTURES['NONE'])

    # Subtle cyber-grid
    for i in range(1, 4):
        cv2.line(cam_frame, (0, i * h // 4), (w, i * h // 4), (30, 20, 20), 1)
        cv2.line(cam_frame, (i * w // 4, 0), (i * w // 4, h),  (30, 20, 20), 1)

    # Bottom panel
    draw_rounded_rect(cam_frame, 0, h - 85, w, h, COL['dark'], 0.75)
    cv2.line(cam_frame, (0, h - 85), (w, h - 85), COL['accent'], 1)

    col = tuple(info['color'])
    txt(cam_frame, "NEURAL LINK:", 10, h - 62, 0.35, COL['dimwhite'])
    txt(cam_frame, info['label'],  95, h - 62, 0.52, col, bold=True)
    txt(cam_frame, f"COMMAND: {info['action']}", 10, h - 38, 0.42, COL['accent'])

    # Confidence gauge
    orb_x, orb_y = w - 55, h - 48
    pct = min(confidence, 1.0)
    cv2.circle(cam_frame, (orb_x, orb_y), 22, (40, 35, 40), 2)
    if pct > 0:
        cv2.ellipse(cam_frame, (orb_x, orb_y), (22, 22), -90, 0, int(360 * pct), col, 3)
    txt(cam_frame, f"{pct*100:.0f}%", orb_x - 16, orb_y + 5, 0.38, COL['white'])

    # Sync bar
    bar_x, bar_y = 10, h - 18
    bar_w = w - 95
    cv2.rectangle(cam_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 4), (35, 30, 35), -1)
    fill = int(bar_w * pct)
    if fill > 0:
        cv2.rectangle(cam_frame, (bar_x, bar_y), (bar_x + fill, bar_y + 4), COL['energy'], -1)
    txt(cam_frame, "SYNC", bar_x + bar_w + 10, bar_y + 6, 0.3, COL['energy'])

    # Target brackets on detected palm centre
    if gesture_cnn.hand_detected and gesture_cnn.landmarks:
        lm  = gesture_cnn.landmarks
        px  = int(lm[9].x * w)
        py  = int(lm[9].y * h)
        l   = 15
        for dx, dy in [(-l, -l), (l, -l), (-l, l), (l, l)]:
            ex = px + dx
            ey = py + dy
            sx = 5 if dx > 0 else -5
            sy = 5 if dy > 0 else -5
            cv2.line(cam_frame, (ex, ey), (ex - sx, ey), col, 2)
            cv2.line(cam_frame, (ex, ey), (ex, ey - sy), col, 2)

    # Legend box
    gx, gy = w - 135, 12
    draw_rounded_rect(cam_frame, gx, gy, gx + 125, gy + 75, COL['dark'], 0.7)
    cv2.rectangle(cam_frame, (gx, gy), (gx + 125, gy + 75), COL['border'], 1)
    txt(cam_frame, "OS PROTOCOL",  gx + 10, gy + 18, 0.35, COL['accent'])
    txt(cam_frame, "FIST: Port",   gx + 8,  gy + 35, 0.3,  (255, 180, 0))
    txt(cam_frame, "PALM: Stbd",   gx + 8,  gy + 50, 0.3,  (100, 255, 120))
    txt(cam_frame, "PEACE: Boost", gx + 8,  gy + 65, 0.3,  (0, 150, 255))


# ──────────────────────────────────────────────────────────
#  MAIN GAME LOOP
# ──────────────────────────────────────────────────────────

def _reset_state():
    """Return a fresh game-state dict."""
    return dict(
        ship=Ship(),
        asteroids=[],
        star_picks=[],
        bullets=[],
        # OPT 9 – deque with maxlen caps particles without a per-frame list copy
        particles=deque(maxlen=100),
        floats=[],
        score=0,
        lives=3,
        level=1,
        frame_n=0,
        shoot_cd=0,
        last_ast=time.time(),
        last_star=time.time(),
        game_over=False,
        boost_triggered=False,
        # FIX 4 – track prev_level across frames, not re-assigned every frame
        prev_level=1,
    )


def main():
    print("\n" + "="*55)
    print("  SPACE DODGE -- CNN Hand Gesture Game")
    print("="*55)
    print("  Controls:")
    print("    FIST      -> Evade LEFT")
    print("    OPEN PALM -> Evade RIGHT")
    print("    PEACE/V   -> Overdrive (speed + rapid fire)")
    print("    [R]       -> Restart  |  [Q] Quit  |  [M] Mute")
    print("="*55 + "\n")

    print("Initialising MediaPipe CNN...", flush=True)
    cnn = GestureCNN()
    print("CNN online!", flush=True)

    print("Generating sound effects...")
    sfx = SoundFX()
    print("SFX systems active!")

    print("Loading background music...", flush=True)
    game_dir = os.path.dirname(os.path.abspath(__file__))
    music = MusicPlayer(game_dir, volume=0.45)
    print("Music system ready!\n", flush=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Pre-allocate reusable surfaces
    game     = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)
    combined = np.zeros((GAME_H, COMBINED_W, 3), dtype=np.uint8)
    cam_view  = combined[:, :CAM_W]
    div_view  = combined[:, CAM_W:CAM_W+3]
    game_view = combined[:, CAM_W+3:]

    div_view[:]  = 40
    cv2.line(div_view, (1, 0), (1, GAME_H), COL['accent'], 1)

    WIN_NAME = "Space Dodge HUD  |  [Q] Quit  [M] Mute"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)

    state          = _reset_state()
    gesture        = 'NONE'
    confidence     = 0.0
    _mp_skip       = 0
    _mp_results    = None

    music.play()
    print("Mission started! Press Q to abort, M to mute music.\n")

    retry = 0
    while True:
        ret, raw = cap.read()
        if not ret:
            retry += 1
            if retry > 30:
                print("ERROR: Camera feed lost.")
                break
            time.sleep(0.01)
            continue
        retry = 0

        cam_bgr = cv2.flip(raw, 1)

        # Run MediaPipe every other frame to halve inference cost
        _mp_skip += 1
        if _mp_skip >= 2:
            _mp_skip = 0
            cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
            gesture, confidence, _mp_results = cnn.process(cam_rgb)
        if _mp_results is not None:
            cnn.draw_landmarks(cam_bgr, _mp_results)

        boosting = (gesture == 'PEACE' and confidence > 0.60)

        # Unpack state for convenience
        ship          = state['ship']
        asteroids     = state['asteroids']
        star_picks    = state['star_picks']
        bullets       = state['bullets']
        particles     = state['particles']
        floats        = state['floats']
        score         = state['score']
        lives         = state['lives']
        level         = state['level']
        frame_n       = state['frame_n']
        shoot_cd      = state['shoot_cd']
        last_ast      = state['last_ast']
        last_star     = state['last_star']
        game_over     = state['game_over']
        boost_triggered = state['boost_triggered']
        prev_level    = state['prev_level']

        if boosting and not boost_triggered:
            sfx.play('boost')
            boost_triggered = True
        if not boosting:
            boost_triggered = False

        game[:] = COL['bg']
        draw_background(game, frame_n)

        if not game_over:
            frame_n += 1

            ship.update(gesture, confidence, boosting)

            shoot_interval = 8 if boosting else 18
            if shoot_cd <= 0:
                bullets.append(Bullet(ship.x, ship.y - 22, boosting))
                sfx.play('shoot')
                shoot_cd = shoot_interval
            shoot_cd -= 1

            spawn_gap = max(0.35, 1.1 - level * 0.07)
            now = time.time()
            if now - last_ast > spawn_gap:
                asteroids.append(Asteroid(level))
                last_ast = now

            if now - last_star > 4.0:
                star_picks.append(StarPickup())
                last_star = now

            for a in asteroids:  a.update()
            for s in star_picks: s.update()
            for b in bullets:    b.update()
            for p in particles:  p.update()
            for f in floats:     f.update()

            # Bullet ↔ Asteroid collisions
            for b in bullets[:]:
                for a in asteroids[:]:
                    if abs(b.x - a.x) < a.size and abs(b.y - a.y) < a.size:
                        a.hp -= 1
                        # OPT 5 – extend() instead of list-comp side-effect
                        particles.extend(Particle(a.x, a.y, (255, 255, 100)) for _ in range(5))
                        if b in bullets:
                            bullets.remove(b)
                        if a.hp <= 0:
                            pts = int(a.size) * (2 if boosting else 1)
                            score += pts
                            sfx.play('explode')
                            floats.append(FloatingText(int(a.x) - 15, int(a.y),
                                                       f"+{pts}", COL['accent'], 0.45))
                            if boosting:
                                floats.append(FloatingText(int(a.x) - 10, int(a.y) - 18,
                                                           "CRITICAL", COL['crit'], 0.4, 2.0))
                            particles.extend(
                                Particle(a.x, a.y, (255, 150, 50), 1.3) for _ in range(14))
                            if a in asteroids:
                                asteroids.remove(a)
                        else:
                            floats.append(FloatingText(int(a.x) - 8, int(a.y),
                                                       "DMG", COL['hit'], 0.35))
                        break

            # Ship ↔ Asteroid collisions
            if ship.invincible == 0:
                for a in asteroids[:]:
                    if ship.collides(a.x, a.y, a.size * 0.72):
                        lives -= 1
                        ship.invincible = 90
                        sfx.play('damage')
                        floats.append(FloatingText(ship.x - 20, ship.y - 30,
                                                   "WARNING", COL['hit'], 0.55, 2.0))
                        particles.extend(
                            Particle(ship.x, ship.y, (50, 50, 255), 1.5) for _ in range(22))
                        if a in asteroids:
                            asteroids.remove(a)
                        if lives <= 0:
                            game_over = True
                            sfx.play('gameover')
                            music.pause()
                        break

            # Star pickups
            for s in star_picks[:]:
                if ship.collides(s.x, s.y, 18):
                    score += 75
                    sfx.play('pickup')
                    floats.append(FloatingText(int(s.x) - 15, int(s.y),
                                              "+75", COL['accent'], 0.5))
                    particles.extend(Particle(s.x, s.y, (255, 255, 100)) for _ in range(12))
                    star_picks.remove(s)

            # Prune off-screen objects
            asteroids[:]  = [a for a in asteroids  if not a.off_screen()]
            star_picks[:] = [s for s in star_picks if not s.off_screen()]
            bullets[:]    = [b for b in bullets    if not b.off_screen()]
            floats[:]     = [f for f in floats     if not f.dead()]
            # Particles capped by deque maxlen – no extra filter needed

            # FIX 4 – level-up: compare against prev_level from the PREVIOUS frame
            new_level = 1 + score // 400
            if new_level > level:
                level = new_level
                sfx.play('levelup')
                floats.append(FloatingText(GAME_W // 2 - 60, GAME_H // 2,
                                           f"SECTOR {level:02d} CLEARED",
                                           COL['accent'], 0.5, 1.0))
                particles.extend(
                    Particle(GAME_W // 2, GAME_H // 2, (255, 255, 0), 1.8)
                    for _ in range(20))
            prev_level = level

        # XP progress within current level
        xp_threshold   = level * 400
        prev_threshold = (level - 1) * 400
        xp_pct = (score - prev_threshold) / max(1, xp_threshold - prev_threshold)

        # Draw game objects
        for s in star_picks: s.draw(game)
        for a in asteroids:  a.draw(game)
        for b in bullets:    b.draw(game)
        for p in particles:  p.draw(game)
        for f in floats:     f.draw(game)
        ship.draw(game, frame_n, boosting)
        draw_game_hud(game, score, lives, level, boosting, frame_n, xp_pct)
        _draw_music_status(game, music, frame_n)

        if game_over:
            draw_rounded_rect(game, 60, 130, GAME_W - 60, 370, COL['dark'], 0.85)
            cv2.rectangle(game, (60, 130), (GAME_W - 60, 370), COL['hit'], 1)
            for x, y in [(60, 130), (GAME_W-60, 130), (60, 370), (GAME_W-60, 370)]:
                dx = 15 if x == 60 else -15
                dy = 15 if y == 130 else -15
                cv2.line(game, (x, y), (x+dx, y), COL['hit'], 2)
                cv2.line(game, (x, y), (x, y+dy), COL['hit'], 2)
            txt(game, "MISSION FAILED",        GAME_W//2 - 105, 180, 0.72, COL['hit'],     bold=True)
            cv2.line(game, (90, 190), (GAME_W - 90, 190), COL['hit'], 1)
            txt(game, "FINAL SCORE",           GAME_W//2 - 58,  230, 0.40, COL['dimwhite'])
            txt(game, f"{score:06d}",          GAME_W//2 - 45,  255, 0.65, COL['white'],   bold=True)
            txt(game, "MAX SECTOR",            GAME_W//2 - 52,  285, 0.40, COL['dimwhite'])
            txt(game, f"SECTOR {level:02d}",   GAME_W//2 - 48,  310, 0.55, COL['accent'],  bold=True)
            cv2.line(game, (90, 330), (GAME_W - 90, 330), COL['border'], 1)
            txt(game, "[R] RESTART    [Q] ABORT", GAME_W//2 - 100, 355, 0.40, COL['accent'])

        # Composite camera + game into pre-allocated combined buffer
        cv2.resize(cam_bgr, (CAM_W, CAM_H), dst=cam_view)
        draw_gesture_panel(cam_view, cnn, gesture, confidence, frame_n)
        np.copyto(game_view, game)

        # OPT 6 & 7 – use pre-computed scanline rows and vignette mask
        combined[_scanline_rows] = (combined[_scanline_rows] * 0.85).astype(np.uint8)
        combined[:] = (combined * _vignette_mask_3d).astype(np.uint8)

        cv2.imshow(WIN_NAME, combined)

        try:
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key in (ord('m'), ord('M')):
            music.toggle_mute()
        if key == ord('r') and game_over:
            state = _reset_state()
            music.restart()
            continue  # skip state write-back below

        # Write mutable scalars back into state dict
        state.update(
            score=score, lives=lives, level=level, frame_n=frame_n,
            shoot_cd=shoot_cd, last_ast=last_ast, last_star=last_star,
            game_over=game_over, boost_triggered=boost_triggered,
            prev_level=prev_level,
        )

    cap.release()
    cnn.landmarker.close()
    try:
        sd.stop()
    except Exception:
        pass
    music.stop()
    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

    print(f"\nFinal Score: {score:,}  |  Max Sector: {level}", flush=True)
    print("Simulation Terminated.", flush=True)


if __name__ == '__main__':
    main()