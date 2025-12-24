---
layout: default
title: "06: Gesture Control"
parent: Projects
nav_order: 6
permalink: /projects/06-gesture-control
---

# Project 6: Gesture-Controlled Presentation System

Control PowerPoint/PDF presentations using hand gestures - perfect for touchless presenting.

## What You'll Learn

1. **Hand Detection** - Using MediaPipe
2. **Landmark Extraction** - 21 hand keypoints
3. **Gesture Recognition** - Finger states and movements
4. **Screen Automation** - pyautogui integration
5. **Gesture Debouncing** - Smooth interaction

## Supported Gestures

| Gesture | Action |
|---------|--------|
| Swipe Left | Next slide |
| Swipe Right | Previous slide |
| Open Palm | Pause/Black screen |
| Fist | Resume |
| Thumbs Up | Volume up |
| Peace Sign | Volume down |

## Usage

```bash
# Demo mode (visualization only)
python main.py --demo

# Full control mode
python main.py --control
```

## Hand Landmarks

MediaPipe detects 21 landmarks per hand:
```
        8   12  16  20
        |   |   |   |
    4   7   11  15  19
    |   |   |   |   |
    3   6   10  14  18
    |   |   |   |   |
    2   5   9   13  17
    |    \  |  /   /
    1      \|/   /
    |       0---/
    WRIST
```

## Gesture Detection Logic

### Finger State Detection
```python
# Finger is extended if tip is above pip joint
is_extended = landmarks[tip]['y'] < landmarks[pip]['y']
```

### Swipe Detection
```python
# Track position history
# Detect significant horizontal movement
if abs(dx) > threshold:
    return 'swipe_right' if dx > 0 else 'swipe_left'
```

## Key Dependencies

| Library | Purpose |
|---------|---------|
| MediaPipe | Hand detection and landmarks |
| pyautogui | Keyboard/mouse automation |
| OpenCV | Image processing and display |

## Code Highlights

### Finger States
```python
# Finger tip and pip (second joint) indices
tips = [4, 8, 12, 16, 20]
pips = [3, 6, 10, 14, 18]

# Check if tip is above pip (finger extended)
for tip, pip in zip(tips[1:], pips[1:]):
    fingers.append(lm[tip]['y'] < lm[pip]['y'])
```

### Gesture Cooldown
```python
# Prevent repeated triggers
if current_time - last_action_time < cooldown:
    return None  # Skip this gesture
```

## Real-World Applications

- Touchless presentations
- Accessibility interfaces
- Smart home control
- Gaming input
- Sign language recognition

## Performance Tips

1. Use `max_hands=1` for better performance
2. Implement gesture cooldown
3. Use position history for swipe detection
4. Flip camera for mirror effect

## References

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html)
- [pyautogui Documentation](https://pyautogui.readthedocs.io/)
