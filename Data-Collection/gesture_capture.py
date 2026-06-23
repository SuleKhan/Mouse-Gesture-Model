from pynput.mouse import Button, Controller

mouse = Controller()

# Read pointer position
while True:
    print('The current pointer position is {}'.format(
        mouse.position))
    