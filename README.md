# 🚀 Space Dodge — CNN Hand Gesture RPG Game

A gesture-controlled space shooter with a **classic RPG-style UI**, built with **MediaPipe CNN** and **OpenCV**.

Dodge asteroids, collect gold, and level up through zones — all controlled by hand gestures detected in real-time via your webcam.

---

## ⚡ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy sounddevice pygame
   ```

2. **Download the MediaPipe Model** (if not included):
   ```
   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```
   Place `hand_landmarker.task` in the project root.

3. **Run the Game**:
   ```bash
   python space_dodge.py
   ```

---

## 🕹️ Controls

| Gesture | In-Game Action | Keyboard |
| :--- | :--- | :--- |
| ✊ **Fist** | Move **LEFT** | — |
| 🖐 **Open Palm** | Move **RIGHT** | — |
| ✌ **Peace / V** | **Overdrive** (Speed + Rapid Fire + Shield) | — |
| — | Retry (on Game Over) | `[R]` |
| — | Quit | `[Q]` / `[ESC]` |
| — | Mute / Unmute Music | `[M]` |

---

## 🎮 RPG-Style Game UI

The HUD is designed with a **classic RPG aesthetic**:

- **❤️ HP Hearts** — 3 heart icons showing remaining lives (filled = alive, outline = lost)
- **💰 Gold Counter** — Score displayed as gold with a coin icon
- **⚔️ Level Badge** — `LV.XX` in an ornamental RPG window with double-border
- **📊 EXP Bar** — Blue experience bar tracking progress to the next zone
- **🛡️ Status Window** — Right-side panel showing current zone and shield status
- **⚡ Overdrive Banner** — RPG "Limit Break" styled activation banner
- **🎯 Spells Panel** — Webcam overlay showing gesture-to-action mappings in an RPG window
- **🏆 Game Over Screen** — "You have fallen..." dialog with Gold Earned and Zone Reached stats
- **🌟 Floating Combat Text** — `+25 Gold`, `CRITICAL!`, `WARNING!`, `HIT` popups

### Visual Design
- **Double-border RPG windows** with golden corner ornaments
- **Warm fantasy color palette** — deep purples, golden accents, warm UI tones
- **Anti-aliased text** for crisp, readable HUD elements
- **Vignette post-processing** for cinematic framing
- **Twinkling starfield** with nebula glow background

---

## 🛠️ Technical Features

### 🧠 CNN Gesture Recognition
Uses a multi-stage **Convolutional Neural Network** (MediaPipe Hand Landmarker) to track 21 hand keypoints in real-time. Gesture classification analyzes finger extension states and landmark geometry to detect Fist, Open Palm, and Peace signs with >90% accuracy.

### 🔊 Procedural Audio
- **SFX**: All sound effects (lasers, explosions, pickups, level-ups) are synthesized procedurally using `sounddevice` — no external audio files required.
- **BGM**: Integrated music player via `pygame.mixer` that auto-detects and loops any `.mp3`/`.wav`/`.ogg` file in the project folder.

### 🎨 Rendering & Performance
- **Particle System**: Dynamic particles for thrusters, explosions, and level-up effects.
- **Pre-computed Vignette**: Vignette mask computed once at startup, reused every frame.
- **Efficient Compositing**: Camera + game surfaces composited into a single pre-allocated buffer.
- **Deque-capped Particles**: `collections.deque(maxlen=100)` for O(1) particle management.
- **MediaPipe Skip-frame**: CNN inference runs every other frame to halve GPU cost.

### ⚙️ Difficulty Scaling
- Asteroid spawn rate and speed increase with each zone.
- Overdrive mode grants rapid fire, speed boost, shield, and 2× gold multiplier.

---

## 📁 Project Structure

| File | Description |
| :--- | :--- |
| `space_dodge.py` | Main game — all logic, rendering, and AI in a single file |
| `hand_landmarker.task` | MediaPipe CNN model for hand landmark detection |
| `*.mp3 / .wav / .ogg` | (Optional) Background music — drop any audio file here |

---

## 🎓 Academic Context

This project demonstrates the application of **CNN-based Computer Vision** in a real-time interactive environment, showcasing the pipeline from raw webcam frames → hand landmark detection → gesture classification → game control, all running at 30 FPS with a polished RPG-themed interface.
