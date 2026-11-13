# M4 threat model

One threat per level, two mitigations. Written as a design; the broker settings are in `m4/mosquitto.conf` and have not been run.

| Level | Threat | Mitigation applied |
|---|---|---|
| Device | Someone with the Pi can read credentials or swap the model file | Credentials only in environment variables (not in the repo); model path is read-only in the container (`m3/Dockerfile`) |
| Network | A device on the LAN publishes fake `detection` messages or reads the topics | Anonymous access off, password file, TLS-only listener on 8883 (`m4/mosquitto.conf`) |
| Application | A forged or stale message makes the actuator or the PLC act | Debounce before any action (`m3/decide.py`); fail-safe release if frames stop |
| Cloud / dashboard | Dashboard exposed to the internet | Dashboard binds to the local network only (Vizion `pi/web/`); no cloud service is used |
