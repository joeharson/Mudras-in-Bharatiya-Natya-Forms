# Real-Time Mudra Detection System

A sophisticated web application that detects and recognizes classical Indian hand gestures (mudras) in real-time using computer vision technology.

## 🌟 Features

- **Real-time Detection**: Instant recognition of hand gestures using webcam
- **Extensive Mudra Support**:
  - 16 single-hand mudras
  - 11 two-handed mudras
- **Live Visualization**:
  - Real-time video feed
  - Hand landmark overlay
  - Gesture recognition feedback
- **Detailed Information**:
  - Mudra names and meanings
  - Traditional context and usage

## 🚀 Technical Stack

- **Backend**:
  - Python 3.x
  - Flask web framework
  - OpenCV (cv2)
  - MediaPipe
- **Frontend**:
  - HTML5
  - CSS3
  - JavaScript
  - Server-Sent Events

## 📋 Prerequisites

- Python 3.7 or higher
- Webcam
- Modern web browser (Chrome recommended)

## 🔧 Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd main
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app3.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## 🎯 Supported Mudras

### Single-Hand Mudras
- Pataka (Flag)
- Tripataka (Three-fingered Flag)
- Ardhapataka (Half Flag)
- Kartarimukha (Scissors)
- Mayura (Peacock)
- And many more...

### Two-Hand Mudras
- Anjali (Prayer)
- Pushpanjali (Flower Offering)
- Namaskara (Greeting)
- And more...

## 💻 Usage

1. Start the application
2. Position your hand(s) in front of the webcam
3. Form a mudra gesture
4. View real-time detection results
5. Read the meaning and context

## 🔍 Detection Details

- Uses 21 hand landmarks for tracking
- Supports both single and dual-hand gestures
- Real-time distance-based calculations
- Gesture confidence scoring

## 🌐 Browser Compatibility

- Google Chrome (recommended)
- Firefox
- Microsoft Edge
- Safari


## 🙏 Acknowledgments

- MediaPipe team for hand tracking solution
- Classical dance practitioners for mudra validation
- Open source community for various tools and libraries



---
Made with ❤️ for preserving classical Indian dance forms through technology
