# M4 PLC link (design, not wired)

No PLC was available. The intended mapping, if a Modbus-capable PLC were attached:

| Source | PLC tag | Type |
|---|---|---|
| Debounced state of `sct013` (`m3/decide.py`) | `DETECT_SCT013` | BOOL coil |
| Detection confidence x 100 | `DETECT_CONF` | INT holding register |
| Fail-safe (no frames for 1 s) | `VISION_FAULT` | BOOL coil |

The Python side would write the coil with `pymodbus` on every state change. Nothing here has been run. The closest working output in this work is the GPIO signal on the Pi in the Vizion repository (`pi/leds.py`).
