/**
 * ESP32 Robot Arm Controller Configuration
 *
 * Pin mappings, calibration values, and network settings
 * for the 4-DOF robot arm with MG90S servos.
 */

#ifndef CONFIG_H
#define CONFIG_H

// ============== Network Configuration ==============
#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#define WEBSOCKET_PORT 81
#define MDNS_NAME "robot-arm"

// ============== Servo Pin Mappings ==============
// Adjust these based on your wiring
#define SERVO_BASE_PIN 2      // Base rotation (joint 0)
#define SERVO_SHOULDER_PIN 3  // Shoulder (joint 1)
#define SERVO_ELBOW_PIN 4     // Elbow (joint 2)
#define SERVO_GRIPPER_PIN 5   // Gripper (joint 3)

// ============== Joystick Pin Mappings ==============
#define JOYSTICK_X_PIN 0      // ADC pin for X axis
#define JOYSTICK_Y_PIN 1      // ADC pin for Y axis
#define JOYSTICK_BTN_PIN 6    // Digital pin for button
#define JOYSTICK2_X_PIN 2     // Second joystick X (for gripper)
#define JOYSTICK2_Y_PIN 3     // Second joystick Y

// ============== Servo Calibration ==============
// Min/Max pulse width in microseconds for MG90S servos
#define SERVO_MIN_US 500
#define SERVO_MAX_US 2400

// Joint angle limits (degrees)
struct JointLimits {
    float min_angle;
    float max_angle;
    float home_angle;
};

const JointLimits JOINT_LIMITS[4] = {
    {0.0, 180.0, 90.0},    // Base: full rotation
    {30.0, 150.0, 90.0},   // Shoulder: limited range
    {30.0, 150.0, 90.0},   // Elbow: limited range
    {10.0, 90.0, 45.0}     // Gripper: open/close
};

// ============== Control Parameters ==============
#define CONTROL_LOOP_HZ 50           // Main control loop frequency
#define TELEMETRY_HZ 20              // State reporting frequency
#define JOYSTICK_DEADZONE 50         // ADC deadzone (0-4095)
#define JOYSTICK_SENSITIVITY 0.5     // Degrees per joystick unit
#define SMOOTHING_FACTOR 0.3         // Low-pass filter coefficient

// ============== Recording Mode ==============
#define RECORDING_BUFFER_SIZE 1000   // Max timesteps per episode
#define AUTO_SAVE_INTERVAL_MS 30000  // Auto-save every 30 seconds

// ============== Safety Limits ==============
#define MAX_VELOCITY_DEG_S 90.0      // Maximum joint velocity
#define EMERGENCY_STOP_PIN 7         // Optional e-stop button

#endif // CONFIG_H
