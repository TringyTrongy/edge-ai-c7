"""Show the broker refuses a client with no password and accepts one with password + TLS.
Usage: MQTT_HOST=... MQTT_USER=... MQTT_PASS=... MQTT_CA=certs/ca.crt python m4/mqtt_check.py
Expected: step 1 'refused' (CONNACK not authorised or TLS error), step 2 'connected', step 3 message received.
"""
import os, ssl, time
import paho.mqtt.client as mqtt

H = os.environ.get("MQTT_HOST", "localhost")


def attempt(user, pw, tls):
    rc = {}
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="c7-check")
    if user: c.username_pw_set(user, pw)
    if tls: c.tls_set(ca_certs=os.environ["MQTT_CA"], tls_version=ssl.PROTOCOL_TLSv1_2)
    c.on_connect = lambda cl, u, f, reason, p: rc.update(reason=str(reason))
    try:
        c.connect(H, 8883 if tls else 1883, 10); c.loop_start(); time.sleep(2); c.loop_stop()
    except Exception as e:
        rc["reason"] = f"error: {e}"
    return rc.get("reason", "no answer")


print("1 no password, no TLS :", attempt(None, None, False))
print("2 password + TLS      :", attempt(os.environ["MQTT_USER"], os.environ["MQTT_PASS"], True))
