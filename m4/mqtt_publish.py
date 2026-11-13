"""Publish the debounced detection state to MQTT over TLS with a username and password.
Credentials come from the environment, never from the repo: MQTT_USER, MQTT_PASS, MQTT_HOST, MQTT_CA.
"""
import os, json, time, ssl
import paho.mqtt.client as mqtt

TOPIC = "vizion/c7/detection"


def client():
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="c7-publisher")
    c.username_pw_set(os.environ["MQTT_USER"], os.environ["MQTT_PASS"])
    c.tls_set(ca_certs=os.environ["MQTT_CA"], tls_version=ssl.PROTOCOL_TLSv1_2)
    c.will_set("vizion/c7/status", json.dumps({"online": False}), retain=True)
    c.connect(os.environ.get("MQTT_HOST", "localhost"), 8883, 30)
    c.loop_start()
    return c


def publish_state(c, cls, conf, state):
    c.publish(TOPIC, json.dumps({"t": time.time(), "cls": cls, "conf": conf, "state": state}), qos=1)
