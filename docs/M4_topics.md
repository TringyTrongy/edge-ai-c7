# M4 MQTT topics

| Topic | Payload (JSON) | Rate | Meaning |
|---|---|---|---|
| `vizion/c7/detection` | `{"t": 1760000000.12, "cls": "sct013", "conf": 0.91, "state": "on"}` | on change | Debounced detection state from `m3/decide.py` |
| `vizion/c7/anomaly` | `{"t": ..., "score": 0.83, "limit": 0.5, "alert": true}` | 1 Hz | Motor anomaly score (camera-based, from the Vizion repo `pi/motor_monitor/`) |
| `vizion/c7/status` | `{"t": ..., "fps": 14.7, "uptime_s": 3600}` | 0.1 Hz | Health and speed, retained, with a last-will of `{"online": false}` |

`m4/mqtt_publish.py` publishes the first topic; `m4/mqtt_check.py` subscribes to it. Neither has been run against a broker yet.
