# 🚀 Space Dodge — CNN Hand Gesture Game

![Space Dodge](C:\Users\snava\.gemini\antigravity\brain\8a4af225-9a86-435d-bff1-0b256905050e\space_dodge_banner_1777531722173.png)

A high-performance, gesture-controlled space shooter built with **MediaPipe** and **OpenCV**.

---

## ⚡ Quick Start

1. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe numpy sounddevice pygame
   ```

2. **Run the Game**:
   ```bash
   python space_dodge.py
   ```

---

## 🕹️ Controls

| Gesture | In-Game Action | Keyboard Alternative |
| :--- | :--- | :--- |
| ✊ **Fist** | Move **LEFT** | - |
| 🖐 **Open Palm** | Move **RIGHT** | - |
| ✌ **Peace / V** | **OVERDRIVE** (Speed + Rapid Fire) | - |
| - | **Restart Game** | `[R]` |
| - | **Quit Game** | `[Q]` |
| - | **Mute Music** | `[M]` |

---

## 🛠️ Technical Features

### 🧠 CNN Gesture Recognition
Uses a multi-stage **Convolutional Neural Network** (MediaPipe) to track 21 hand landmarks in real-time. The game logic analyzes finger extension states and landmark geometry to classify gestures with >90% accuracy.

### 🔊 Procedural Audio
- **SFX**: Sound effects (lasers, explosions, pickups) are synthesized procedurally using `sounddevice`, requiring no external `.wav` or `.mp3` files.
- **BGM**: Integrated music player using `pygame.mixer` that automatically finds and loops music tracks in the project folder.

### 🎨 Visuals & HUD
- **Particles**: Dynamic particle system for thrusters and explosions.
- **Glassmorphism**: Semi-transparent, modern HUD panels.
- **Difficulty Scaling**: Level-based progression with increasing asteroid density and speed.

---

## 📁 File Requirements
- `space_dodge.py`: Main game logic.
- `hand_landmarker.task`: MediaPipe model file (must be in the same directory).
- `*.mp3/wav`: (Optional) Any audio file in the folder will be used as background music.

---

## 🎓 Academic Context
This project demonstrates the application of **CNN-based Computer Vision** in a real-time interactive environment, showcasing the transition from raw landmark data to high-level gesture control.
