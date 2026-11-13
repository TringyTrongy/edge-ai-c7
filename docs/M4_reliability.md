# M4 reliability

| Requirement | Where it stands |
|---|---|
| Auto-restart after a crash | `m3/docker-compose.yml` sets `restart: unless-stopped` (not run). The Pi autostart launchers and safety shutdowns are in the Vizion repository (`pi/vizion-*.desktop`) |
| Drift report | Not done |
| OTA v1 to v2 with rollback | Not done |
