/**
 * ESP32 Robot Arm Controller - Main Entry Point
 *
 * Implements:
 * - WebSocket server for host PC communication
 * - Servo control for 4-DOF arm
 * - Joystick teleoperation input
 * - Real-time state streaming for data collection
 *
 * Part of the ESP32 Robot Learning from Demonstration project
 * https://github.com/goker/esp32-robot-lfd
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include <ESPmDNS.h>
#include "config.h"

// ============== Global Objects ==============
WebSocketsServer webSocket = WebSocketsServer(WEBSOCKET_PORT);
Servo servos[4];

// ============== State Variables ==============
struct RobotState {
    float joint_positions[4];      // Current joint angles (degrees)
    float joint_targets[4];        // Target joint angles
    float gripper_state;           // 0.0 = open, 1.0 = closed
    uint32_t timestamp_ms;
    bool is_recording;
    bool is_autonomous;
};

struct TeleopInput {
    float joystick_axes[4];        // Normalized joystick values (-1 to 1)
    bool buttons[2];               // Button states
};

RobotState robot_state;
TeleopInput teleop_input;
uint8_t connected_client = 255;    // Track connected client

// ============== Timing ==============
unsigned long last_control_time = 0;
unsigned long last_telemetry_time = 0;
const unsigned long control_period_us = 1000000 / CONTROL_LOOP_HZ;
const unsigned long telemetry_period_ms = 1000 / TELEMETRY_HZ;

// ============== Function Declarations ==============
void setupWiFi();
void setupServos();
void setupJoysticks();
void readJoysticks();
void updateServos();
void sendTelemetry();
void handleWebSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length);
void processCommand(uint8_t num, JsonDocument& doc);
float mapJoystickToAngle(int raw_value, int joint_idx);
float clampAngle(float angle, int joint_idx);
float smoothValue(float current, float target, float factor);

// ============== Setup ==============
void setup() {
    Serial.begin(115200);
    Serial.println("\n=== ESP32 Robot Arm Controller ===");
    Serial.println("Initializing...");

    // Initialize state
    for (int i = 0; i < 4; i++) {
        robot_state.joint_positions[i] = JOINT_LIMITS[i].home_angle;
        robot_state.joint_targets[i] = JOINT_LIMITS[i].home_angle;
    }
    robot_state.gripper_state = 0.5;
    robot_state.is_recording = false;
    robot_state.is_autonomous = false;

    setupWiFi();
    setupServos();
    setupJoysticks();

    // Start WebSocket server
    webSocket.begin();
    webSocket.onEvent(handleWebSocketEvent);

    // Setup mDNS
    if (MDNS.begin(MDNS_NAME)) {
        Serial.printf("mDNS responder started: %s.local\n", MDNS_NAME);
        MDNS.addService("ws", "tcp", WEBSOCKET_PORT);
    }

    Serial.println("Setup complete. Ready for connections.");
    Serial.printf("WebSocket: ws://%s:%d\n", WiFi.localIP().toString().c_str(), WEBSOCKET_PORT);
}

// ============== Main Loop ==============
void loop() {
    webSocket.loop();

    unsigned long now = micros();
    unsigned long now_ms = millis();

    // Control loop (50 Hz)
    if (now - last_control_time >= control_period_us) {
        last_control_time = now;

        // Read joystick input (only in teleoperation mode)
        if (!robot_state.is_autonomous) {
            readJoysticks();

            // Update targets from joystick
            for (int i = 0; i < 4; i++) {
                float delta = teleop_input.joystick_axes[i] * JOYSTICK_SENSITIVITY;
                robot_state.joint_targets[i] = clampAngle(
                    robot_state.joint_targets[i] + delta, i
                );
            }
        }

        // Smooth movement and update servos
        updateServos();
    }

    // Telemetry loop (20 Hz)
    if (now_ms - last_telemetry_time >= telemetry_period_ms) {
        last_telemetry_time = now_ms;
        robot_state.timestamp_ms = now_ms;
        sendTelemetry();
    }
}

// ============== WiFi Setup ==============
void setupWiFi() {
    Serial.printf("Connecting to WiFi: %s\n", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\nFailed to connect. Starting AP mode...");
        WiFi.softAP("RobotArm-Setup", "robotarm123");
        Serial.printf("AP IP: %s\n", WiFi.softAPIP().toString().c_str());
    }
}

// ============== Servo Setup ==============
void setupServos() {
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);

    int pins[4] = {SERVO_BASE_PIN, SERVO_SHOULDER_PIN, SERVO_ELBOW_PIN, SERVO_GRIPPER_PIN};

    for (int i = 0; i < 4; i++) {
        servos[i].setPeriodHertz(50);
        servos[i].attach(pins[i], SERVO_MIN_US, SERVO_MAX_US);
        servos[i].write(robot_state.joint_positions[i]);
    }

    Serial.println("Servos initialized");
}

// ============== Joystick Setup ==============
void setupJoysticks() {
    pinMode(JOYSTICK_BTN_PIN, INPUT_PULLUP);
    // ADC pins don't need explicit setup on ESP32
    Serial.println("Joysticks initialized");
}

// ============== Read Joysticks ==============
void readJoysticks() {
    int raw_values[4];
    raw_values[0] = analogRead(JOYSTICK_X_PIN);
    raw_values[1] = analogRead(JOYSTICK_Y_PIN);
    raw_values[2] = analogRead(JOYSTICK2_X_PIN);
    raw_values[3] = analogRead(JOYSTICK2_Y_PIN);

    for (int i = 0; i < 4; i++) {
        // Center is ~2048, map to -1 to 1 with deadzone
        int centered = raw_values[i] - 2048;
        if (abs(centered) < JOYSTICK_DEADZONE) {
            teleop_input.joystick_axes[i] = 0.0;
        } else {
            teleop_input.joystick_axes[i] = centered / 2048.0;
        }
    }

    teleop_input.buttons[0] = !digitalRead(JOYSTICK_BTN_PIN);  // Active low
}

// ============== Update Servos ==============
void updateServos() {
    for (int i = 0; i < 4; i++) {
        // Smooth movement
        robot_state.joint_positions[i] = smoothValue(
            robot_state.joint_positions[i],
            robot_state.joint_targets[i],
            SMOOTHING_FACTOR
        );

        // Write to servo
        servos[i].write((int)robot_state.joint_positions[i]);
    }

    // Update gripper state (normalized 0-1)
    robot_state.gripper_state = (robot_state.joint_positions[3] - JOINT_LIMITS[3].min_angle) /
                                 (JOINT_LIMITS[3].max_angle - JOINT_LIMITS[3].min_angle);
}

// ============== Send Telemetry ==============
void sendTelemetry() {
    if (connected_client == 255) return;  // No client connected

    StaticJsonDocument<512> doc;

    doc["type"] = "state";
    doc["timestamp_ms"] = robot_state.timestamp_ms;

    JsonArray joint_pos = doc.createNestedArray("joint_positions");
    JsonArray joint_tgt = doc.createNestedArray("joint_targets");
    JsonArray teleop = doc.createNestedArray("teleop_input");

    for (int i = 0; i < 4; i++) {
        joint_pos.add(robot_state.joint_positions[i]);
        joint_tgt.add(robot_state.joint_targets[i]);
        teleop.add(teleop_input.joystick_axes[i]);
    }

    doc["gripper"] = robot_state.gripper_state;
    doc["recording"] = robot_state.is_recording;
    doc["autonomous"] = robot_state.is_autonomous;

    String output;
    serializeJson(doc, output);
    webSocket.sendTXT(connected_client, output);
}

// ============== WebSocket Event Handler ==============
void handleWebSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {
        case WStype_DISCONNECTED:
            Serial.printf("[%u] Disconnected\n", num);
            if (num == connected_client) {
                connected_client = 255;
                robot_state.is_autonomous = false;  // Safety: stop autonomous mode
            }
            break;

        case WStype_CONNECTED: {
            IPAddress ip = webSocket.remoteIP(num);
            Serial.printf("[%u] Connected from %s\n", num, ip.toString().c_str());
            connected_client = num;

            // Send initial state
            sendTelemetry();
            break;
        }

        case WStype_TEXT: {
            StaticJsonDocument<256> doc;
            DeserializationError error = deserializeJson(doc, payload, length);

            if (error) {
                Serial.printf("JSON parse error: %s\n", error.c_str());
                return;
            }

            processCommand(num, doc);
            break;
        }

        default:
            break;
    }
}

// ============== Process Commands ==============
void processCommand(uint8_t num, JsonDocument& doc) {
    const char* cmd = doc["cmd"];

    if (strcmp(cmd, "set_targets") == 0) {
        // Set joint target positions (from policy inference)
        JsonArray targets = doc["targets"];
        for (int i = 0; i < 4 && i < targets.size(); i++) {
            robot_state.joint_targets[i] = clampAngle(targets[i], i);
        }
    }
    else if (strcmp(cmd, "start_recording") == 0) {
        robot_state.is_recording = true;
        Serial.println("Recording started");
    }
    else if (strcmp(cmd, "stop_recording") == 0) {
        robot_state.is_recording = false;
        Serial.println("Recording stopped");
    }
    else if (strcmp(cmd, "set_autonomous") == 0) {
        robot_state.is_autonomous = doc["value"] | false;
        Serial.printf("Autonomous mode: %s\n", robot_state.is_autonomous ? "ON" : "OFF");
    }
    else if (strcmp(cmd, "home") == 0) {
        // Move to home position
        for (int i = 0; i < 4; i++) {
            robot_state.joint_targets[i] = JOINT_LIMITS[i].home_angle;
        }
        Serial.println("Moving to home position");
    }
    else if (strcmp(cmd, "get_state") == 0) {
        sendTelemetry();
    }
}

// ============== Utility Functions ==============
float clampAngle(float angle, int joint_idx) {
    return constrain(angle, JOINT_LIMITS[joint_idx].min_angle, JOINT_LIMITS[joint_idx].max_angle);
}

float smoothValue(float current, float target, float factor) {
    return current + factor * (target - current);
}
