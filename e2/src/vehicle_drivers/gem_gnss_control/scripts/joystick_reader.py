import struct
import threading

class RawJoystickReader:
    def __init__(self, device="/dev/input/js0"):
        self.device = device
        self.running = True
        self.axes = [0.0] * 8
        self.buttons = [0] * 12
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        EVENT_SIZE = struct.calcsize("llHHI")
        try:
            with open(self.device, "rb") as js:
                while self.running:
                    evbuf = js.read(EVENT_SIZE)
                    if not evbuf:
                        continue
                    _, value, type_, number, _ = struct.unpack("llHHI", evbuf)
                    if type_ == 2:  # axis
                        if number < len(self.axes):
                            self.axes[number] = value / 32767.0
                    elif type_ == 1:  # button
                        if number < len(self.buttons):
                            self.buttons[number] = value
        except FileNotFoundError:
            print(f"[controller_reader] Device {self.device} not found.")

    def get_state(self):
        return self.axes, self.buttons
