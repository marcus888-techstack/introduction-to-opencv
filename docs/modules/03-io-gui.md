---
layout: default
title: "03: I/O & GUI"
parent: Modules
nav_order: 3
permalink: /modules/03-io-gui
---

# Module 3: I/O and GUI
{: .fs-9 }

Reading, writing, and displaying images and videos with OpenCV's I/O and GUI facilities.
{: .fs-6 .fw-300 }

---

## Topics Covered

- Reading and writing images
- Video capture and writing
- Display windows and keyboard handling
- Trackbars and mouse events
- Drawing functions

---

## Algorithm Explanations

### 1. Image Reading (imread)

**Read Modes**:

| Flag | Value | Description |
|:-----|:------|:------------|
| `IMREAD_COLOR` | 1 | Load as BGR (default) |
| `IMREAD_GRAYSCALE` | 0 | Load as single channel |
| `IMREAD_UNCHANGED` | -1 | Load with alpha channel |

**Decoding Process**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                     Image Reading Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐     ┌──────────┐     ┌────────────┐     ┌──────────┐ │
│   │  File   │────▶│ Decoder  │────▶│ Raw Pixels │────▶│  NumPy   │ │
│   │ (disk)  │     │(JPEG/PNG)│     │  (BGR)     │     │  Array   │ │
│   └─────────┘     └──────────┘     └────────────┘     └──────────┘ │
│                                                                     │
│   photo.jpg        Decompress      [B,G,R,B,G,R,    shape: (H,W,3) │
│   image.png        & Decode         B,G,R,B,G,R]    dtype: uint8   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 2. Video Capture

**VideoCapture** handles reading from video files, cameras, and network streams.

**Video Source Types**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                     Video Capture Sources                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────────┐                                                 │
│   │  Video File   │──────┐                                          │
│   │ "video.mp4"   │      │                                          │
│   └───────────────┘      │      ┌─────────────────┐                 │
│                          ├─────▶│  VideoCapture   │────▶ Frames    │
│   ┌───────────────┐      │      │                 │                 │
│   │    Camera     │──────┤      └─────────────────┘                 │
│   │   device: 0   │      │                                          │
│   └───────────────┘      │                                          │
│                          │                                          │
│   ┌───────────────┐      │                                          │
│   │ Network Stream│──────┘                                          │
│   │"rtsp://..."   │                                                 │
│   └───────────────┘                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Video Properties**:

| Property | ID | Description |
|:---------|:---|:------------|
| `CAP_PROP_FRAME_WIDTH` | 3 | Frame width |
| `CAP_PROP_FRAME_HEIGHT` | 4 | Frame height |
| `CAP_PROP_FPS` | 5 | Frames per second |
| `CAP_PROP_FRAME_COUNT` | 7 | Total frames |

---

### 3. Video Writing

**FourCC (Four Character Code)**:
```
FourCC = 4 ASCII characters identifying the codec

  ┌───┬───┬───┬───┐
  │ X │ V │ I │ D │  = XVID codec (MPEG-4)
  └───┴───┴───┴───┘

  ┌───┬───┬───┬───┐
  │ M │ J │ P │ G │  = Motion JPEG
  └───┴───┴───┴───┘
```

| Code | Format | Description |
|:-----|:-------|:------------|
| `XVID` | AVI | MPEG-4 codec |
| `mp4v` | MP4 | MPEG-4 Part 2 |
| `MJPG` | AVI | Motion JPEG |

---

### 4. Trackbars

**Trackbar Visualization**:
```
┌─────────────────────────────────────────────────────────────────────┐
│  Window: "Controls"                                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Brightness ├────────────────●───────────────────┤ 150 / 255      │
│                              ▲                                      │
│                              │                                      │
│                         Slider position                             │
│                                                                     │
│   Contrast   ├───●────────────────────────────────┤  30 / 100      │
│                                                                     │
│   Threshold  ├──────────────────────────────●─────┤ 200 / 255      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 5. Mouse Events

**Event Types**:

| Event | Description |
|:------|:------------|
| `EVENT_MOUSEMOVE` | Mouse moved |
| `EVENT_LBUTTONDOWN` | Left button pressed |
| `EVENT_LBUTTONUP` | Left button released |
| `EVENT_RBUTTONDOWN` | Right button pressed |
| `EVENT_LBUTTONDBLCLK` | Left double-click |

---

### 6. Drawing Functions

**Drawing Primitives**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                     Drawing Functions                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   LINE                   RECTANGLE              CIRCLE              │
│                                                                     │
│   pt1 ●                  pt1 ●─────────┐        ┌───────┐          │
│        ╲                     │         │       ╱    ●    ╲         │
│         ╲                    │         │      │  center   │         │
│          ╲                   │         │      │  ●───────│radius    │
│           ● pt2              └─────────● pt2   ╲         ╱         │
│                                                  └───────┘          │
│                                                                     │
│   ELLIPSE                POLYLINES              TEXT                │
│                                                                     │
│       ╭──────╮           ●─────●               ┌─────────────┐     │
│      ╱        ╲           ╲   ╱                │ Hello World │     │
│     │   axes   │           ╲ ╱                 └─────────────┘     │
│      ╲  a × b ╱             ●                                      │
│       ╰──────╯                                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Line Types**:

| Type | Description |
|:-----|:------------|
| `LINE_8` | 8-connected (default) |
| `LINE_4` | 4-connected |
| `LINE_AA` | Anti-aliased |

---

## Tutorial Files

| File | Description |
|:-----|:------------|
| `01_image_io.py` | imread, imwrite, formats, encoding |
| `02_video_io.py` | VideoCapture, VideoWriter, camera input |
| `03_gui_basics.py` | Windows, keyboard, trackbars, mouse events, drawing |

---

## Key Functions Reference

| Function | Description |
|:---------|:------------|
| `cv2.imread(path, flags)` | Load image |
| `cv2.imwrite(path, img)` | Save image |
| `cv2.VideoCapture(src)` | Open video/camera |
| `cv2.VideoWriter(...)` | Create video writer |
| `cv2.namedWindow(name)` | Create window |
| `cv2.imshow(name, img)` | Display image |
| `cv2.waitKey(delay)` | Wait for key |
| `cv2.createTrackbar(...)` | Create slider |
| `cv2.setMouseCallback(...)` | Set mouse handler |
| `cv2.line(...)` | Draw line |
| `cv2.rectangle(...)` | Draw rectangle |
| `cv2.circle(...)` | Draw circle |
| `cv2.putText(...)` | Draw text |

---

## Further Reading

- [GUI Features](https://docs.opencv.org/4.x/dc/d4d/tutorial_py_table_of_contents_gui.html)
- [Image Codecs](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html)
- [Video I/O](https://docs.opencv.org/4.x/dd/de7/group__videoio.html)
