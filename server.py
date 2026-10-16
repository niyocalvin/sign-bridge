"""Live fingerspelling server. The browser (on the laptop) finds the hand with MediaPipe and sends a 224x224 hand crop here;
this PC classifies it with the fine-tuned SigLIP2 on the GPU and applies the stability rule.

Run (sign-model venv has the GPU torch):
    C:\\Users\\calvi\\sign-model\\.venv\\Scripts\\python.exe live\\server.py [--model models\\final_siglip2] [--port 8100]
Open https://<this-PC-address>:8100 on the laptop (accept the certificate warning once; the browser needs HTTPS for the camera).
"""
import argparse
import base64
import datetime
import io
import ipaddress
import json
import mimetypes
import os
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.ao.quantization
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import AutoModelForImageClassification

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT / "src"))                 # motion.py lives in live/ on the Space, in src/ on this PC
from debounce import Debouncer  # noqa: E402
from motion import build_model as build_motion, window_features  # noqa: E402
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")
# Settings from environment variables (so the same file runs on this PC and on a Hugging Face Space):
#   FS_DEVICE=cpu            force CPU (default: GPU when available)
#   FS_MODELS='{"name": "folder or Hugging Face repo id"}'   models to load (default: the two local ones)
#   FS_ALLOWED_ORIGINS=https://user.github.io,...             pages allowed to call this server (default: any)
#   FS_ACCESS_CODE=secret    if set, every request must carry ?code=secret
DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("FS_DEVICE") != "cpu" else "cpu"
ORIGINS = [o.strip() for o in os.environ.get("FS_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
ACCESS_CODE = os.environ.get("FS_ACCESS_CODE", "")
COLLECT = os.environ.get("FS_COLLECT", "1") == "1"                       # the page's "Collect samples" panel saves recordings here
COLLECT_DIR = Path(os.environ.get("FS_COLLECT_DIR", ROOT / "data" / "collected"))
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])
M = {}


def allowed(code, origin=None):
    if ACCESS_CODE and code != ACCESS_CODE:
        return False
    return ORIGINS == ["*"] or origin is None or origin in ORIGINS


def classify(jpeg: bytes, name=None):
    t0 = time.perf_counter()
    model, id2 = M["models"][name if name in M["models"] else M["default"]]
    im = Image.open(io.BytesIO(jpeg)).convert("RGB").resize((224, 224), Image.BICUBIC)
    x = torch.from_numpy((np.asarray(im, dtype=np.float32) / 255 - 0.5) / 0.5).permute(2, 0, 1)[None].to(DEVICE)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        p = model(pixel_values=x).logits.float().softmax(-1)[0].cpu().numpy()
    order = np.argsort(-p)[:5]
    top = [[id2[int(i)], round(float(p[i]), 4)] for i in order]
    return top, (time.perf_counter() - t0) * 1000


def motion_probs(seq):
    """Last 2 s of hand points from the page -> {'J':p,'Z':p,'other':p} (or None when there are too few frames)."""
    w, h = seq["video"]
    P = np.array(seq["lm"], float).reshape(-1, 21, 3) * np.array([w, h, w])
    f = window_features(np.array(seq["t"], float), P, float(seq["end"]), W=2000.0)
    if f is None:
        return None
    with torch.inference_mode():
        p = M["motion"](torch.from_numpy(f[None])).softmax(-1)[0].numpy()
    return {"J": round(float(p[0]), 4), "Z": round(float(p[1]), 4), "other": round(float(p[2]), 4)}


@app.get("/")
def index():
    page = HERE / "index.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"service": "fingerspelling model server", "models": list(M["models"]), "device": DEVICE,
                         "use": "open the web page with ?server=<this address>"})


@app.get("/models")
def models(code: str = ""):
    if not allowed(code):
        return JSONResponse({"error": "access code needed"}, status_code=401)
    return {"models": list(M["models"]), "default": M["default"], "device": DEVICE, "needs_code": bool(ACCESS_CODE), "collect": COLLECT, "motion": "motion" in M}


@app.post("/collect")
async def collect(request: Request, label: str, code: str = ""):
    """Saves one recording from the page's 'Collect samples' panel: per frame the hand points (+ timestamp) and the 224x224 hand crop (JPEG).
    Local PC only (FS_COLLECT=1); stored in data/collected/<label>/<time>/ . Nothing leaves this PC."""
    if not COLLECT:
        return JSONResponse({"error": "collecting is switched off on this server"}, status_code=403)
    if not allowed(code):
        return JSONResponse({"error": "access code needed"}, status_code=401)
    if not re.fullmatch(r"[A-Z]|other|word", label):
        return JSONResponse({"error": "label must be A-Z, other or word"}, status_code=400)
    body = await request.json()
    d = COLLECT_DIR / label / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    d.mkdir(parents=True, exist_ok=True)
    frames, n_hand = [], 0
    for i, f in enumerate(body["frames"]):
        entry = {"t": round(f["t"], 1), "hand": f.get("hand")}
        if f.get("hand") and f.get("jpg"):
            (d / f"{i:03d}.jpg").write_bytes(base64.b64decode(f["jpg"]))
            entry.update({"lm": f["lm"], "box": f["box"], "file": f"{i:03d}.jpg"})
            n_hand += 1
        frames.append(entry)
    meta = {"label": label, "video": body.get("video"), "secs": body.get("secs"), "frames": frames}
    if label == "word":                                  # word test: the word that was meant and the text the page typed
        meta.update({"expected": str(body.get("expected", ""))[:40], "typed": str(body.get("typed", ""))[:80]})
    (d / "frames.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"saved": True, "hand_frames": n_hand, "total_frames": len(frames)}


@app.post("/predict")
async def predict(request: Request, model: str = None, code: str = ""):
    if not allowed(code):
        return JSONResponse({"error": "access code needed"}, status_code=401)
    top, ms = classify(await request.body(), model)
    return JSONResponse({"top": top, "ms": round(ms, 1)})


@app.websocket("/ws")
async def ws(sock: WebSocket):
    if not allowed(sock.query_params.get("code", ""), sock.headers.get("origin")):
        await sock.close(code=1008)
        return
    await sock.accept()
    deb = Debouncer(n=8, adaptive=True)
    current = None
    try:
        while True:
            msg = await sock.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text") is not None:                      # control messages
                c = json.loads(msg["text"])
                if "seq" in c:                                   # hand-point history for the J/Z motion model
                    await sock.send_json({"motion": motion_probs(c["seq"]) if "motion" in M else None})
                    continue
                if "n" in c:
                    deb.n = int(c["n"])
                if "min_conf" in c:
                    deb.min_conf = float(c["min_conf"])
                if "adaptive" in c:
                    deb.adaptive = bool(c["adaptive"])
                if "model" in c:
                    current = c["model"]
                    deb.reset()
                if c.get("nohand"):
                    prog, commit = deb.update(None, 0)
                    await sock.send_json({"top": [], "progress": prog, "commit": commit, "ms": 0})
                if c.get("reset"):
                    deb.reset()
                continue
            top, ms = classify(msg["bytes"], current)
            prog, commit = deb.update(top[0][0], top[0][1])
            await sock.send_json({"top": top, "progress": prog, "commit": commit, "ms": round(ms, 1)})
    except WebSocketDisconnect:
        pass


if (HERE / "static").exists():
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


def lan_addresses():
    ips = set()
    try:
        ips.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def ensure_certificate(cert_dir: Path):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    cert_dir.mkdir(exist_ok=True)
    cert, key, marker = cert_dir / "cert.pem", cert_dir / "key.pem", cert_dir / "ips.json"
    ips = lan_addresses()
    if cert.exists() and key.exists() and marker.exists() and json.loads(marker.read_text()) == ips:
        return cert, key
    pk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    host = socket.gethostname()
    san = [x509.DNSName("localhost"), x509.DNSName(host), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    san += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"{host} fingerspelling (self-signed)")])
    now = datetime.datetime.now(datetime.timezone.utc)
    crt = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(pk.public_key())
           .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
           .not_valid_after(now + datetime.timedelta(days=825))
           .add_extension(x509.SubjectAlternativeName(san), critical=False).sign(pk, hashes.SHA256()))
    key.write_bytes(pk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    cert.write_bytes(crt.public_bytes(serialization.Encoding.PEM))
    marker.write_text(json.dumps(ips))
    return cert, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8100)))
    ap.add_argument("--no-https", action="store_true", default=bool(os.environ.get("FS_NO_HTTPS")))
    a = ap.parse_args()
    if DEVICE == "cpu":
        torch.set_num_threads(int(os.environ.get("FS_THREADS", os.cpu_count() or 2)))
    # name shown in the page -> folder or Hugging Face repo id. "your webcam" = public data + the user's own webcam frames
    candidates = json.loads(os.environ["FS_MODELS"]) if os.environ.get("FS_MODELS") else {
        "your webcam (public + your data)": str(ROOT / "models" / "final_siglip2_crop"), "public data only": str(ROOT / "models" / "final_siglip2")}
    M["models"] = {}
    for name, src in candidates.items():
        if Path(src).exists() and not (Path(src) / "config.json").exists():
            continue
        m = AutoModelForImageClassification.from_pretrained(src).to(DEVICE).eval()
        if DEVICE == "cpu":
            try:
                m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
            except Exception as e:
                print(f"Dynamic quantization note: {e}")
        M["models"][name] = (m, {int(k): v for k, v in m.config.id2label.items()})
    M["default"] = next(iter(M["models"]))
    for name in M["models"]:
        classify(_blank(), name)                                 # warm-up
    motion_file = Path(os.environ.get("FS_MOTION", ROOT / "models" / "motion_jz.pt"))
    if motion_file.exists():                                     # small J/Z movement model (runs on the CPU, ~1 ms)
        mm = build_motion()
        mm.load_state_dict(torch.load(motion_file, map_location="cpu"))
        M["motion"] = mm.eval()
    ssl, scheme = {}, "http"
    if not a.no_https:
        cert, key = ensure_certificate(HERE / "certs")
        ssl, scheme = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}, "https"
    print(f"\nmodels {list(M['models'])} on {torch.cuda.get_device_name() if DEVICE == 'cuda' else 'CPU'}\nOpen on the laptop (same Wi-Fi/network):")
    for ip in lan_addresses():
        print(f"    {scheme}://{ip}:{a.port}")
    print(f"    {scheme}://localhost:{a.port}   (this PC)\nFirst visit: accept the certificate warning.\n", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning", **ssl)


def _blank():
    b = io.BytesIO()
    Image.new("RGB", (224, 224), (128, 128, 128)).save(b, "JPEG")
    return b.getvalue()


if __name__ == "__main__":
    main()
