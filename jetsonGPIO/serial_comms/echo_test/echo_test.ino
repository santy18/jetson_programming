// Arduino UNO (CH340): talks over USB Serial at 115200
// - Echoes any line it receives
// - Sends a heartbeat every 1s

unsigned long lastBeat = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial) { /* no-op for UNO but ok */ }

  Serial.println("ARDUINO_READY");
}

void loop() {
  // Read a line from Jetson and echo back
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.length() > 0) {
      Serial.print("ECHO: ");
      Serial.println(line);

      if (line == "PING") {
        Serial.println("PONG");
      }
    }
  }

  // Periodic heartbeat
  if (millis() - lastBeat >= 1000) {
    lastBeat = millis();
    Serial.println("HEARTBEAT");
  }
}
