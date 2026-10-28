"""Render a synthetic detection dataset from the Fusion 360 STL exports.

Classes: 0 = sct013 (IDRP A6 current transformer), 1 = robot (Fusion robot chassis).
Every image is rendered on the PC (software OpenGL), so the boxes come from the object masks,
not from hand labelling. Split 70/20/10 is made per image index before rendering, with a
different random seed per image, so no image or pose is shared between splits.
"""
import os, sys, json, math, random
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, trimesh, pyrender, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {
    0: ("sct013", os.path.join(ROOT, "cad", "SCT-013-000_current_transformer.stl"), [0.12, 0.12, 0.13]),
    1: ("robot", os.path.join(ROOT, "cad", "robot_base_link.stl"), [0.25, 0.45, 0.75]),
}
W, H = 640, 480
N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
OUT = os.path.join(ROOT, "data")


def load(path):
    m = trimesh.load(path)
    m.apply_translation(-m.bounds.mean(axis=0))
    m.apply_scale(1.0 / np.abs(m.bounds).max() / 2.0)  # longest half-extent -> 0.5
    return m


def rot(rng):
    q = np.array([rng.gauss(0, 1) for _ in range(4)])
    q /= np.linalg.norm(q)
    return trimesh.transformations.quaternion_matrix(q)


def background(rng):
    kind = rng.choice(["grad", "noise", "wood", "grid", "flat"])
    base = np.array([rng.randint(40, 235) for _ in range(3)], dtype=np.float32)
    if kind == "flat":
        img = np.ones((H, W, 3), np.float32) * base
    elif kind == "grad":
        a = np.linspace(0, 1, W)[None, :, None] if rng.random() < .5 else np.linspace(0, 1, H)[:, None, None]
        img = base * (0.6 + 0.6 * a) * np.ones((H, W, 3), np.float32)
    elif kind == "noise":
        n = np.random.RandomState(rng.randint(0, 1 << 30)).rand(H // 8, W // 8, 3).astype(np.float32)
        img = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC) * 120 + base * 0.6
    elif kind == "wood":
        y = np.arange(H)[:, None]
        s = np.sin(y / rng.uniform(2, 6) + np.random.RandomState(rng.randint(0, 1 << 30)).rand(1, W) * 3) * 12
        img = (base + s[..., None]) * np.ones((H, W, 3), np.float32)
    else:
        img = np.ones((H, W, 3), np.float32) * base
        step = rng.randint(24, 60)
        img[::step, :, :] *= 0.6
        img[:, ::step, :] *= 0.6
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(rng.randint(0, 6)):  # distractor shapes
        c = tuple(int(x) for x in np.random.RandomState(rng.randint(0, 1 << 30)).randint(0, 255, 3))
        p = (rng.randint(0, W), rng.randint(0, H))
        if rng.random() < .5:
            cv2.circle(img, p, rng.randint(8, 60), c, -1)
        else:
            cv2.rectangle(img, p, (p[0] + rng.randint(10, 120), p[1] + rng.randint(10, 90)), c, -1)
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0, 2.5))


def main():
    renderer = pyrender.OffscreenRenderer(W, H)
    meshes = {k: load(v[1]) for k, v in MODELS.items()}
    idx = list(range(N))
    random.Random(7).shuffle(idx)
    cut1, cut2 = int(N * .7), int(N * .9)
    split_of = {i: ("train" if r < cut1 else "val" if r < cut2 else "test") for r, i in enumerate(idx)}
    for s in ("train", "val", "test"):
        os.makedirs(f"{OUT}/images/{s}", exist_ok=True)
        os.makedirs(f"{OUT}/labels/{s}", exist_ok=True)
    manifest = []
    for i in range(N):
        rng = random.Random(1000 + i)
        sp = split_of[i]
        img = background(rng).astype(np.float32)
        labels = []
        n_obj = rng.choice([0, 1, 1, 1, 2, 2, 3]) if rng.random() > 0.06 else 0
        for _ in range(n_obj):
            cls = rng.choice([0, 1])
            mesh = meshes[cls]
            scene = pyrender.Scene(ambient_light=[0.25] * 3, bg_color=[0, 0, 0, 0])
            mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=MODELS[cls][2] + [1.0],
                                                     metallicFactor=rng.uniform(0, .4), roughnessFactor=rng.uniform(.35, .9))
            pm = pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=False)
            scene.add(pm, pose=rot(rng))
            d = rng.uniform(1.6, 3.6)
            cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.2)
            pose = np.eye(4)
            pose[2, 3] = d
            ax = rng.uniform(-.5, .5)
            R = trimesh.transformations.rotation_matrix(ax, [1, 0, 0])
            scene.add(cam, pose=R @ pose)
            L = trimesh.transformations.rotation_matrix(rng.uniform(-1, 1), [rng.random(), rng.random(), rng.random()])
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=rng.uniform(2, 6)), pose=L)
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=rng.uniform(0, 2)), pose=np.eye(4))
            rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            a = rgba[..., 3]
            ys, xs = np.where(a > 8)
            if len(xs) < 200:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            sc = rng.uniform(0.35, 0.95) * min(H / bh, W / bw) * 0.7
            obj = cv2.resize(rgba[y0:y1 + 1, x0:x1 + 1], (max(8, int(bw * sc)), max(8, int(bh * sc))))
            oh, ow = obj.shape[:2]
            if ow >= W - 4 or oh >= H - 4:
                continue
            px, py = rng.randint(0, W - ow), rng.randint(0, H - oh)
            al = obj[..., 3:4].astype(np.float32) / 255.0
            roi = img[py:py + oh, px:px + ow]
            img[py:py + oh, px:px + ow] = roi * (1 - al) + obj[..., :3] * al
            # tight box from the final alpha mask
            yy, xx = np.where(obj[..., 3] > 8)
            bx0, bx1, by0, by1 = px + xx.min(), px + xx.max() + 1, py + yy.min(), py + yy.max() + 1
            labels.append((cls, (bx0 + bx1) / 2 / W, (by0 + by1) / 2 / H, (bx1 - bx0) / W, (by1 - by0) / H))
        out = np.clip(img + np.random.RandomState(i).randn(H, W, 3) * rng.uniform(0, 6), 0, 255).astype(np.uint8)
        name = f"img_{i:04d}"
        cv2.imwrite(f"{OUT}/images/{sp}/{name}.jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(f"{OUT}/labels/{sp}/{name}.txt", "w") as f:
            for l in labels:
                f.write("%d %.6f %.6f %.6f %.6f\n" % l)
        manifest.append({"image": name, "split": sp, "objects": len(labels), "seed": 1000 + i})
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"))
    open(f"{OUT}/data.yaml", "w").write(
        "path: " + OUT + "\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: sct013\n  1: robot\n")
    from collections import Counter
    print(Counter(m["split"] for m in manifest))


if __name__ == "__main__":
    main()
