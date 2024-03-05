# Object Tracking

![tracker.png](docs/tracker.png)

## Installation

The tracking package can be installed using pip by running the following commands:

```bash
cd object-tracking
pip install -e .
```

## Usage
### Interactive tracking example

Run the following lines with Python:

```python
from tracking import InteractiveTracker
InteractiveTracker()
```

### Example usage

Run the following lines with Python:

```python
from tracking import Tracker, Detection

# Initialize the tracker
config = {}
tracker = Tracker(config)

# Get some detections
detections = [Detection(500, 100, 100, 100, "object", 0.9),
              Detection(200, 300, 200, 100, "object", 0.8)]

# Update the tracker
tracker.update(detections)

# Visualize the tracker state
tracker.show(savefig="tracker_1.png")

# Get some other detections
detections = [Detection(480, 120, 140, 120, "object", 0.95),
              Detection(200, 300, 170, 80, "object", 0.9)]

# Update the tracker again
tracker.update(detections)

# Visualize the new tracker state
tracker.show(savefig="tracker_2.png")
```