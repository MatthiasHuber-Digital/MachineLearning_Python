# simple_detector

This is a simple project that uses OpenCV to perform detection.

## Installation:

```bash
pip install -r requirements.txt
```

## Usage:

```python
import Detector

detector = Detector()
img = cv2.imread('path/to/image.jpg')
bboxes = detector.predict(img)
```
