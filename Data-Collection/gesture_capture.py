from PIL import Image, ImageDraw
from pynput import mouse

gesture = "Screenshot" # Replace with the desired gesture name
totalGestures = 100
fixIndex = 99
isRecording = False

def on_move(x, y, injected):
    # print('Pointer moved to {}; it was {}'.format(
    #     (x, y), 'faked' if injected else 'not faked'))
    if x > 1900 and y < 30:
        # Stop listener
        return False
    if isRecording and (x, y) not in mousePath:
        mousePath.append((x, y))

def on_click(x, y, button, pressed, injected):
    global isRecording
    # print('{} {} at {}; it was {}'.format(
    #     button,
    #     'Pressed' if pressed else 'Released',
    #     (x, y), 'faked' if injected else 'not faked'))
    if button == mouse.Button.x2 and pressed:
        isRecording = True
    elif button == mouse.Button.x2 and not pressed:
        isRecording = False
        return False  # Stop listener

def on_scroll(x, y, dx, dy, injected):
    # print('Scrolled {} at {}; it was {}'.format(
    #     'down' if dy < 0 else 'up',
    #     (x, y), 'faked' if injected else 'not faked'))
    pass

for i in range(totalGestures):
# for i in range(fixIndex - 1, fixIndex):
    mousePath = []

    print(f"Gesture {i+1}/{totalGestures}: Please perform the '{gesture}' gesture")
    with mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll) as listener:
        listener.join()
    print(f"Gesture {i+1}/{totalGestures} recorded.")

    # create blank image
    img = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(img)

    # draw lines between points
    draw.line(mousePath, fill="black", width=3)

    img.save(f"Data-Collection/Gesture-Data/{gesture}/{gesture}_mouse_path_{i+1}.png")