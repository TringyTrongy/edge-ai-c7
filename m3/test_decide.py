import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from m3.decide import Debounce, SimulatedActuator, step


def run(seq):
    d, a, ev = Debounce(), SimulatedActuator(), []
    for s in seq:
        ev.append(step(d, a, [{"cls": "sct013"}] if s else []))
    return ev, a


def test_one_bad_frame_does_not_trigger():
    ev, a = run([0, 0, 1, 0, 0, 0])
    assert "on" not in ev and not a.on


def test_n_in_a_row_triggers_once():
    ev, a = run([0, 1, 1, 1, 1, 1])
    assert ev.count("on") == 1 and a.on


def test_single_dropout_does_not_release():
    ev, a = run([1, 1, 1, 0, 1, 1])
    assert ev.count("off") == 0 and a.on


def test_three_misses_release():
    ev, a = run([1, 1, 1, 0, 0, 0])
    assert ev[-1] == "off" and not a.on


def test_other_class_is_ignored():
    d, a = Debounce(), SimulatedActuator()
    for _ in range(5):
        step(d, a, [{"cls": "robot"}])
    assert not a.on


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
