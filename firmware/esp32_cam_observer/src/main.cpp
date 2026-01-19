/**
 * ESP32-CAM Observer Module
 *
 * Streams camera frames for visual observations in robot learning.
 * Provides MJPEG stream over HTTP and frame timestamps for synchronization.
 *
 * Part of the ESP32 Robot Learning from Demonstration project
 * https://github.com/goker/esp32-robot-lfd
 */

#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include <WebServer.h>
#include <ESPmDNS.h>

// ============== Configuration ==============
#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#define STREAM_PORT 80
#define MDNS_NAME "robot-cam"

// ============== Camera Pin Definitions (ESP32-CAM AI-Thinker) ==============
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// For M5Stack Camera, use different pins - uncomment below:
// #define PWDN_GPIO_NUM     -1
// #define RESET_GPIO_NUM    15
// ... (configure based on your M5Stack camera model)

WebServer server(STREAM_PORT);

// ============== Camera Configuration ==============
void setupCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    // Frame size and quality for robot learning
    // QVGA (320x240) is good balance of speed and detail
    if (psramFound()) {
        config.frame_size = FRAMESIZE_QVGA;  // 320x240
        config.jpeg_quality = 12;             // 0-63, lower = better quality
        config.fb_count = 2;
        config.fb_location = CAMERA_FB_IN_PSRAM;
        config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
        config.frame_size = FRAMESIZE_QVGA;
        config.jpeg_quality = 15;
        config.fb_count = 1;
        config.fb_location = CAMERA_FB_IN_DRAM;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        ESP.restart();
    }

    // Optimize sensor settings for robot learning
    sensor_t* s = esp_camera_sensor_get();
    s->set_brightness(s, 0);     // -2 to 2
    s->set_contrast(s, 0);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_whitebal(s, 1);       // 0 = disable, 1 = enable
    s->set_awb_gain(s, 1);       // 0 = disable, 1 = enable
    s->set_wb_mode(s, 0);        // 0-4: auto, sunny, cloudy, office, home
    s->set_exposure_ctrl(s, 1);  // 0 = disable, 1 = enable
    s->set_aec2(s, 0);           // 0 = disable, 1 = enable
    s->set_gain_ctrl(s, 1);      // 0 = disable, 1 = enable
    s->set_agc_gain(s, 0);       // 0 to 30
    s->set_gainceiling(s, (gainceiling_t)0);  // 0 to 6
    s->set_bpc(s, 0);            // 0 = disable, 1 = enable
    s->set_wpc(s, 1);            // 0 = disable, 1 = enable
    s->set_raw_gma(s, 1);        // 0 = disable, 1 = enable
    s->set_lenc(s, 1);           // 0 = disable, 1 = enable
    s->set_hmirror(s, 0);        // 0 = disable, 1 = enable
    s->set_vflip(s, 0);          // 0 = disable, 1 = enable
    s->set_dcw(s, 1);            // 0 = disable, 1 = enable

    Serial.println("Camera initialized");
}

// ============== MJPEG Stream Handler ==============
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %lu\r\n\r\n";

void handleStream() {
    WiFiClient client = server.client();

    Serial.println("Stream client connected");
    client.println("HTTP/1.1 200 OK");
    client.printf("Content-Type: %s\r\n", STREAM_CONTENT_TYPE);
    client.println("Access-Control-Allow-Origin: *");
    client.println();

    while (client.connected()) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("Frame capture failed");
            continue;
        }

        unsigned long timestamp = millis();

        client.print(STREAM_BOUNDARY);
        client.printf(STREAM_PART, fb->len, timestamp);
        client.write(fb->buf, fb->len);

        esp_camera_fb_return(fb);

        if (!client.connected()) break;
    }

    Serial.println("Stream client disconnected");
}

// ============== Single Frame Handler ==============
void handleCapture() {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        server.send(500, "text/plain", "Camera capture failed");
        return;
    }

    server.sendHeader("Content-Type", "image/jpeg");
    server.sendHeader("Content-Length", String(fb->len));
    server.sendHeader("X-Timestamp", String(millis()));
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);

    esp_camera_fb_return(fb);
}

// ============== Status Handler ==============
void handleStatus() {
    String json = "{";
    json += "\"status\":\"ok\",";
    json += "\"timestamp\":" + String(millis()) + ",";
    json += "\"resolution\":\"320x240\",";
    json += "\"fps\":15,";
    json += "\"psram\":" + String(psramFound() ? "true" : "false");
    json += "}";

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", json);
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
        Serial.println("\nFailed. Starting AP mode...");
        WiFi.softAP("RobotCam-Setup", "robotcam123");
        Serial.printf("AP IP: %s\n", WiFi.softAPIP().toString().c_str());
    }
}

// ============== Setup ==============
void setup() {
    Serial.begin(115200);
    Serial.println("\n=== ESP32-CAM Robot Observer ===");

    setupCamera();
    setupWiFi();

    // Setup mDNS
    if (MDNS.begin(MDNS_NAME)) {
        Serial.printf("mDNS: %s.local\n", MDNS_NAME);
        MDNS.addService("http", "tcp", STREAM_PORT);
    }

    // Setup routes
    server.on("/stream", HTTP_GET, handleStream);
    server.on("/capture", HTTP_GET, handleCapture);
    server.on("/status", HTTP_GET, handleStatus);
    server.on("/", HTTP_GET, []() {
        String html = "<html><body>";
        html += "<h1>Robot Camera Observer</h1>";
        html += "<img src='/stream' style='width:640px;height:480px;'/>";
        html += "<p><a href='/capture'>Single Frame</a> | <a href='/status'>Status</a></p>";
        html += "</body></html>";
        server.send(200, "text/html", html);
    });

    server.begin();
    Serial.printf("HTTP server started on port %d\n", STREAM_PORT);
    Serial.printf("Stream URL: http://%s/stream\n", WiFi.localIP().toString().c_str());
}

// ============== Loop ==============
void loop() {
    server.handleClient();
}
