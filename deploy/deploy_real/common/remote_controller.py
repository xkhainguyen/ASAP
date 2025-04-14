from pynput import keyboard
import threading
import struct

class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        # wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


class KeyListener:
    def __init__(self):
        """Initialize key listener state."""
        self.key_states = {}  # Dictionary to store key states (pressed/released)
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener_thread = threading.Thread(target=self.listener.start, daemon=True)
        self.listener_thread.start()  # Start listening in a separate thread

    def on_press(self, key):
        """Handles key press events and updates the key state."""
        try:
            key_str = key.char if hasattr(key, 'char') else str(key)  # Convert special keys
            self.key_states[key_str] = True
            # print(f"Key {key_str} pressed")
        except AttributeError:
            pass  # Ignore errors for special keys

    def on_release(self, key):
        """Handles key release events and updates the key state."""
        try:
            key_str = key.char if hasattr(key, 'char') else str(key)
            self.key_states[key_str] = False
            # print(f"Key {key_str} released")
        except AttributeError:
            pass  # Ignore errors for special keys

    def is_pressed(self, key):
        """Check if a specific key is currently pressed."""
        return self.key_states.get(key, False)

    def stop_listener(self):
        """Stop the listener when needed."""
        self.listener.stop()
        self.listener_thread.join()