import Jetson.GPIO as GPIO
import time

SERVO_PIN = 32  # physical pin
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz for servo
pwm.start(7.5)  # neutral (~90°)

try:
    pwm.ChangeDutyCycle(5)    # ~0°
    time.sleep(1)
    pwm.ChangeDutyCycle(7.5)  # ~90°
    time.sleep(1)
    pwm.ChangeDutyCycle(10)   # ~180°
    time.sleep(1)
finally:
    pwm.stop()
    GPIO.cleanup()