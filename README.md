# Mouse Gesture Model

A small computer-vision project that recognizes mouse-drawn gestures with a
PyTorch convolutional neural network. The live tool listens for gestures drawn
while holding **Left Ctrl**, classifies the gesture, and prints the predicted
action and confidence.

## Supported gestures

The included model recognizes:

- `Back`
- `Blank-Tab`
- `Close-Tab`
- `Close-Window`
- `Forward`
- `Gmail`
- `Refresh`
- `Screenshot`
- `Wordle`
- `Youtube`

## Project structure

```text
Data-Collection/
  gesture_capture.py       # Capture labeled mouse-path images
  gestures.txt             # Gesture names and intended shortcuts
  Gesture-Data/            # Collected image dataset

Model-Training/
  cnn.py                   # Train the convolutional neural network
  cnn.ipynb                # Notebook version of the training workflow
  gesture_cnn_val_9598.pt  # Included trained model
  gesture_classes.json     # Class labels used by the model
  logits_to_probs.py       # Add probabilities to saved predictions

Mouse-Gesture-Tool/
  tool.py                  # Run live gesture recognition
  tool.ipynb               # Notebook version of the live tool
  cuda_check.py            # Check CUDA availability
```

## Requirements

- Python 3.9 or newer
- A Windows desktop session (the scripts use global mouse and keyboard
  listeners)
- PyTorch and torchvision
- Pillow
- NumPy
- pynput
- Matplotlib (needed by the training script)

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision pillow numpy pynput matplotlib
```

For GPU acceleration, install the PyTorch build appropriate for your CUDA
version from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).
The code automatically uses CUDA when it is available and otherwise falls back
to CPU.

## Run the included model

Run commands from the repository root so the relative model and class-file
paths resolve correctly:

```powershell
python .\Mouse-Gesture-Tool\tool.py
```

Hold **Left Ctrl**, draw a gesture with the mouse, and release **Left Ctrl**.
The predicted class and confidence are printed in the terminal. Press **Esc**
to stop the listener.

The tool writes the most recent captured gesture to
`Mouse-Gesture-Tool/gesture.png`. Move the pointer to the top-right exit area
or press **Esc** to stop the application.

To check whether PyTorch can see a CUDA GPU:

```powershell
python .\Mouse-Gesture-Tool\cuda_check.py
```

## Collect more training data

The dataset uses one subdirectory per class, with one PNG image per gesture.
`gesture_capture.py` is configured through the `gesture` and `totalGestures`
variables near the top of the file.

1. Set `gesture` to the class being recorded.
2. Ensure `Data-Collection/Gesture-Data/<gesture>` exists.
3. Run the collector from the repository root:

   ```powershell
   python .\Data-Collection\gesture_capture.py
   ```

4. For each prompt, press and hold the mouse's forward (`X2`) button while
   drawing the gesture, then release it.

The collector saves 1920x1080 white-background PNGs containing the recorded
mouse path. Keep the gesture style and screen setup reasonably varied so the
model generalizes to new drawings.

## Train the model

`Model-Training/cnn.py` loads the images with `ImageFolder`, creates an 80/20
train/validation split, trains for 30 epochs, and saves the best checkpoint.
Run it from the repository root:

```powershell
python .\Model-Training\cnn.py
```

The script currently writes `gesture_cnn.pt` and `gesture_classes.json` to the
current working directory. To use these newly generated files with
`Mouse-Gesture-Tool/tool.py`, copy them into `Model-Training` using the names
expected by the live tool:

```powershell
Copy-Item .\gesture_cnn.pt .\Model-Training\gesture_cnn_val_9598.pt
Copy-Item .\gesture_classes.json .\Model-Training\gesture_classes.json
```

The training script also displays loss and accuracy plots after training.

## Notes

- The live tool and the training pipeline use slightly different image
  preprocessing based on decisions made during development in attempts to improve accuracy of the model. Ideally, a model should be evaluated with the same kind of input it
  was trained on.
- The included `.pt` checkpoint and gesture dataset are provided for
  experimentation; recognition quality depends on the similarity between
  training and live gestures. For example the model was trained on how I draw the gestures and may have learnt some of my nuances.
- `gestures.txt` records the intended browser actions and shortcuts, but the
  current live tool only performs classification and prints the result. It does
  not automatically execute those shortcuts.
