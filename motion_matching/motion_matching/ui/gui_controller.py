#!/usr/bin/env python3
"""
GUI Keyboard Controller

Communicate with the main program through a file.
"""

import tkinter as tk
import numpy as np
import json
import os
import tempfile
import time
import threading
from typing import Any

# Env var name used to share the per-launch state file path between parent and subprocess.
KEYBOARD_STATE_ENV = "MOTION_MATCHING_KEYBOARD_STATE_FILE"


def _create_state_file() -> str:
    """Securely create a fresh, unique state file via mkstemp(). Returns the path."""
    fd, path = tempfile.mkstemp(prefix="motion_matching_keyboard_state_", suffix=".json")
    os.close(fd)  # We only need the path; controllers reopen by name.
    return path


keyboard_skills_mapping = {
    # Step family
    "1": "step",
    "2": "step_up",
    "3": "step_down",
    # Climb 58 (no high-speed variant)
    "q": "climb_58",
    "w": "climb_58_up",
    "e": "climb_58_down",
    # Climb 76
    "r": "climb_76",
    "t": "climb_76_up",
    "y": "climb_76_down",
    # Climb 94
    "a": "climb_94_up",
    "s": "climb_94_down",
    # Climb 134 (only up, only low-speed)
    "g": "climb_134_up",
    # Roll
    "z": "roll_78",
    "x": "roll_90",
    "c": "roll_136",
    # Vaults & obstacles
    "7": "hurdle",
    "8": "dash_vault",
    "9": "speed_vault",
    "0": "cat_vault",
    # Misc
    "v": "jump_35",
    "b": "drop_92",
}

speed_modifier_keys = {"tab": "toggle_speed"}

key_states_default = {
    "i": False,  # Forward
    "k": False,  # Backward
    "j": False,  # Left
    "l": False,  # Right
}
for key in keyboard_skills_mapping:
    key_states_default[key] = False
for key in speed_modifier_keys:
    key_states_default[key] = False


# File communication keyboard controller
class FileKeyboardController:
    """File communication keyboard controller"""

    def __init__(self) -> None:
        self.running = True
        self.key_states = key_states_default.copy()
        # Securely create a unique state file for this launch (parent owns it).
        self.state_file = _create_state_file()
        self.gui_process = None

    def start(self) -> bool:
        """Start keyboard controller"""
        self.running = True

        # Start GUI process; pass the state file path via env var so child uses the same file.
        import subprocess

        env = os.environ.copy()
        env[KEYBOARD_STATE_ENV] = self.state_file
        try:
            self.gui_process = subprocess.Popen(["python", __file__], env=env)
            print("GUI keyboard controller started")
            print("Please use the pop-up GUI window to control the keyboard")
            print("Press I, K, J, L and other keys to control the character movement")
            print("-" * 50)
            return True
        except Exception as e:
            print(f"Failed to start GUI: {e}")
            print("Will use terminal input as a fallback")
            return self._start_terminal_fallback()

    def _start_terminal_fallback(self) -> bool:
        """Start terminal fallback"""
        print("Terminal keyboard controller started")
        print("Press i, k, j, l keys to control the character movement")
        print("Press 'q' to exit the program")
        print("-" * 50)
        return True

    def stop(self) -> None:
        """Stop keyboard controller"""
        self.running = False
        if self.gui_process:
            self.gui_process.terminate()
            self.gui_process = None
        # Best-effort cleanup of the per-launch state file.
        try:
            os.remove(self.state_file)
        except OSError:
            pass
        print("Keyboard controller stopped")

    def update(self) -> None:
        """Update keyboard state (called in the main loop)"""
        # Try to read GUI state from file
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.key_states = data.get("key_states", self.key_states)
                    gui_running = data.get("running", True)

                    if not gui_running:
                        self.running = False
            except Exception:
                # print(f"[DEBUG] File read error: {e}")
                # If file read fails, use terminal input
                self._handle_terminal_input()
        else:
            # If file does not exist, use terminal input
            self._handle_terminal_input()

    def _handle_terminal_input(self) -> None:
        """Handle terminal input (fallback)"""
        import sys
        import select
        import tty
        import termios

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            try:
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

                if ch.lower() == "q":
                    self.running = False
                    print("\nExit program")
                elif ch.lower() in self.key_states:
                    self.key_states[ch.lower()] = not self.key_states[ch.lower()]
                    status = "Pressed" if self.key_states[ch.lower()] else "Released"
                    print(f"\nKey {ch.upper()}: {status}")

            except (OSError, termios.error, KeyboardInterrupt):
                self.running = False
                print("\nProgram interrupted")

    def is_key_pressed(self, key: str) -> bool:
        """Check if key is pressed"""
        return self.key_states.get(key.lower(), False)

    def get_desired_velocity(self) -> np.ndarray:
        """Return the keyboard-driven unit direction (x forward, y left)."""
        forward = 1.0 if self.is_key_pressed("i") else 0.0
        backward = 1.0 if self.is_key_pressed("k") else 0.0
        left = 1.0 if self.is_key_pressed("j") else 0.0  # J = Left
        right = 1.0 if self.is_key_pressed("l") else 0.0  # L = Right

        return np.array([forward - backward, left - right, 0.0], dtype=np.float32)

    def get_key_status(self) -> dict[str, bool]:
        """Get all key states"""
        return self.key_states.copy()


class GUIKeyboardController:
    def __init__(self) -> None:
        self.running = True
        self.key_states = key_states_default.copy()
        self.root = None
        self.status_labels = {}
        # Latched display state for speed-modifier keys; flipped on each press.
        # The underlying key_states[key] still tracks raw press/release so the
        # main loop in run.py can detect rising edges.
        self.speed_display_state = {key: "Low" for key in speed_modifier_keys}
        # Subprocess: receive the parent's state file path via env var; fallback to fresh mkstemp() if launched standalone.
        self.state_file = os.environ.get(KEYBOARD_STATE_ENV) or _create_state_file()

    def start(self) -> None:
        """Start GUI"""
        self.root = tk.Tk()
        self.root.title("Motion Matching - Keyboard Controller")
        self.root.geometry("700x600")

        # Set window always on top
        self.root.attributes("-topmost", True)

        self._setup_ui()
        self._bind_keys()

        # Start state save thread
        self.save_thread = threading.Thread(target=self._save_state_loop, daemon=True)
        self.save_thread.start()

        print("GUI keyboard controller started")
        self.root.mainloop()

    def _setup_ui(self) -> None:
        """Setup user interface"""
        # Title
        title_label = tk.Label(self.root, text="Motion Matching - Keyboard Controller", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Info
        info_label = tk.Label(
            self.root, text="Press I, K, J, L and other keys to control the character", font=("Arial", 12)
        )
        info_label.pack(pady=5)

        # Key status display: two columns
        self.status_frame = tk.Frame(self.root)
        self.status_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        left_col = tk.Frame(self.status_frame)
        left_col.pack(side=tk.LEFT, anchor=tk.N, padx=20)
        right_col = tk.Frame(self.status_frame)
        right_col.pack(side=tk.LEFT, anchor=tk.N, padx=20)

        left_entries = [
            ("i", "Forward"),
            ("k", "Backward"),
            ("j", "Left"),
            ("l", "Right"),
            *[(key, label) for key, label in speed_modifier_keys.items()],
        ]
        right_entries = [(key, skill) for key, skill in keyboard_skills_mapping.items()]

        for col, entries in [(left_col, left_entries), (right_col, right_entries)]:
            for key, direction in entries:
                frame = tk.Frame(col)
                frame.pack(pady=3, anchor=tk.W)

                label = tk.Label(frame, text=f"{key.upper()}: {direction}", font=("Arial", 11), width=24, anchor=tk.W)
                label.pack(side=tk.LEFT)

                if key in speed_modifier_keys:
                    status_label = tk.Label(frame, text="Low", fg="blue", font=("Arial", 11, "bold"))
                else:
                    status_label = tk.Label(frame, text="Not Pressed", fg="red", font=("Arial", 11, "bold"))
                status_label.pack(side=tk.LEFT, padx=(5, 0))

                self.status_labels[key] = status_label

        # Quit button
        quit_button = tk.Button(
            self.root, text="Quit program", command=self._quit_program, font=("Arial", 12), bg="red", fg="black"
        )
        quit_button.pack(pady=20)

    def _bind_keys(self) -> None:
        """Bind keyboard events"""
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

    def _on_key_press(self, event: Any) -> None:
        """Key press event"""
        key = event.keysym.lower()
        if key in self.key_states and not self.key_states[key]:
            self.key_states[key] = True
            if key in self.status_labels:
                if key in speed_modifier_keys:
                    # Toggle latched display on the rising edge of each press.
                    new_state = "High" if self.speed_display_state[key] == "Low" else "Low"
                    self.speed_display_state[key] = new_state
                    fg = "green" if new_state == "High" else "blue"
                    self.status_labels[key].config(text=new_state, fg=fg)
                else:
                    self.status_labels[key].config(text="Pressed", fg="green")
            # print(f"Key pressed: {key}")
            # Save state immediately
            self._save_state()

    def _on_key_release(self, event: Any) -> None:
        """Key release event"""
        key = event.keysym.lower()
        if key in self.key_states and self.key_states[key]:
            self.key_states[key] = False
            if key in self.status_labels and key not in speed_modifier_keys:
                self.status_labels[key].config(text="Not Pressed", fg="red")
            # print(f"Key released: {key}")
            # Save state immediately
            self._save_state()

    def _quit_program(self) -> None:
        """Quit program"""
        self.running = False
        self.root.quit()

    def _save_state(self) -> None:
        """Save state to file"""
        try:
            state_data = {"key_states": self.key_states, "running": self.running}
            with open(self.state_file, "w") as f:
                json.dump(state_data, f)
            # print(f"[DEBUG] State saved: {self.key_states}")
        except Exception as e:
            print(f"Save state error: {e}")

    def _save_state_loop(self) -> None:
        """Save state to file loop"""
        while self.running:
            try:
                # Save state to JSON file
                state_data = {"key_states": self.key_states, "running": self.running}
                with open(self.state_file, "w") as f:
                    json.dump(state_data, f)
                time.sleep(0.01)  # 10ms update once
            except Exception as e:
                print(f"Save state error: {e}")
                break


def main() -> None:
    controller = GUIKeyboardController()
    controller.start()


if __name__ == "__main__":
    main()
