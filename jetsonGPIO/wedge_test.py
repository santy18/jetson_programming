#!/usr/bin/env python3
"""
Jetson Orin Nano 40-pin GPIO voltage test (Pi Wedge B+ friendly).

- Uses Jetson.GPIO in BOARD mode (physical pin numbers).
- Prints the Pi Wedge B+ "Gxx" label (Raspberry Pi BCM) for each pin.
- Drives each selected pin HIGH for N seconds, then LOW, then moves to next.

How to test with a multimeter:
- Black probe: any GND pin (6, 9, 14, 20, 25, 30, 34, 39)
- Red probe: the pin currently being driven HIGH
- Expect ~3.3V when HIGH, ~0V when LOW
"""

import time
import Jetson.GPIO as GPIO

# ---------------------------
# Configuration
# ---------------------------
HOLD_SECONDS = 10

# Safer, "plain GPIO" style pins (good first set to verify hardware).
# These correspond to common Pi Wedge labels: G4, G17, G27, G22, G5, G6, G12, G13, G19, G26
SAFE_TEST_PINS = [7, 11, 13, 15, 29, 31, 32, 33, 35, 37]

# Full set of GPIO-labeled wedge pins (includes bus pins like I2C/SPI/UART).
# WARNING: Toggling these can interfere with devices/services using those buses.
ALL_WEDGE_GPIO_PINS = [
    3, 5, 7, 8, 10, 11, 12, 13, 15, 16,
    18, 19, 21, 22, 23, 24, 26, 27, 28,
    29, 31, 32, 33, 35, 36, 37, 38, 40
]

# Choose which set to test:
TEST_PINS = SAFE_TEST_PINS  # change to ALL_WEDGE_GPIO_PINS if you really want everything

# ---------------------------
# Pi Wedge "Gxx" (BCM) -> Physical pin map (BOARD)
# ---------------------------
WEDGE_G_TO_PHYS = {
    "G2": 3,
    "G3": 5,
    "G4": 7,
    "G14": 8,
    "G15": 10,
    "G17": 11,
    "G18": 12,
    "G27": 13,
    "G22": 15,
    "G23": 16,
    "G24": 18,
    "G10": 19,
    "G9": 21,
    "G25": 22,
    "G11": 23,
    "G8": 24,
    "G7": 26,
    "G0": 27,   # ID_SD on Pi
    "G1": 28,   # ID_SC on Pi
    "G5": 29,
    "G6": 31,
    "G12": 32,
    "G13": 33,
    "G19": 35,
    "G16": 36,
    "G26": 37,
    "G20": 38,
    "G21": 40,
}

PHYS_TO_WEDGE_G = {v: k for k, v in WEDGE_G_TO_PHYS.items()}


def wedge_label_for_phys(pin: int) -> str:
    return PHYS_TO_WEDGE_G.get(pin, "(no G label)")


def main() -> None:
    GPIO.setmode(GPIO.BOARD)

    print("Selected test pins (physical -> wedge):")
    for p in TEST_PINS:
        print(f"  Pin {p:>2} -> {wedge_label_for_phys(p)}")

    print("\nStarting loop. Measure voltage from the active pin to GND.")
    print("Expect ~3.3V when HIGH, ~0V when LOW.\n")

    try:
        # Set up all pins as outputs LOW initially
        for p in TEST_PINS:
            GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)

        while True:
            for p in TEST_PINS:
                label = wedge_label_for_phys(p)

                GPIO.output(p, GPIO.HIGH)
                print(f"HIGH  for {HOLD_SECONDS}s: Physical Pin {p}  ({label})")
                time.sleep(HOLD_SECONDS)

                GPIO.output(p, GPIO.LOW)
                print(f"LOW   for 1s:         Physical Pin {p}  ({label})\n")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
