"""Smart Aviary Health Monitor (AvianGuard AI) - 4-tab Streamlit demo.

Run:  streamlit run app.py
Expects the Colab outputs in C:\\aviary\\outputs  (override with env var AVIARY_OUT).
"""
import os, json, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

OUT = os.environ.get("AVIARY_OUT", r"C:\aviary\outputs")
BEH = ["Feeding", "Preening", "Swimming", "Walking", "Alert", "Flying", "Resting"]
st.set_page_config(page_title="Smart Aviary Health Monitor", layout="wide")
st.title("Smart Aviary Health Monitor")
st.caption("Classroom benchmark on small public datasets. Not a clinical or veterinary tool.")


def load_json(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def ckpt(name):
    return torch.load(os.path.join(OUT, name), map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- Module 3 (images)
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
IMG_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])


@st.cache_resource
def load_m3(key, n):
    m = models.vgg16(weights=None) if key.startswith("vgg16") else models.alexnet(weights=None)
    m.classifier[6] = nn.Linear(4096, n)
    m.load_state_dict(ckpt(f"m3_{key}.pth"))
    return m.eval()


# ---------------------------------------------------------------- Module 2 (audio)
def cb(i, o, k):
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())


class VGG1D(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.f = nn.Sequential(cb(40, 64, 3), cb(64, 64, 3), nn.MaxPool1d(2), cb(64, 128, 3), cb(128, 128, 3), nn.MaxPool1d(2),
                            cb(128, 256, 3), cb(256, 256, 3), cb(256, 256, 3), nn.MaxPool1d(2), cb(256, 512, 3), cb(512, 512, 3),
                            nn.AdaptiveAvgPool1d(1))
        s.c = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n))

    def forward(s, x):
        return s.c(s.f(x))


class Alex1D(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.f = nn.Sequential(cb(40, 64, 7), nn.MaxPool1d(2), cb(64, 192, 5), nn.MaxPool1d(2), cb(192, 384, 3), cb(384, 256, 3),
                            cb(256, 256, 3), nn.AdaptiveAvgPool1d(1))
        s.c = nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, n))

    def forward(s, x):
        return s.c(s.f(x))


class AudioRNN(nn.Module):
    def __init__(s, n):
        super().__init__()
        s.r = nn.LSTM(40, 128, batch_first=True, num_layers=2, dropout=0.3)
        s.o = nn.Linear(128, n)

    def forward(s, x):
        o, _ = s.r(x.transpose(1, 2))
        return s.o(o[:, -1])


AUDIO_CLS = {"vgg1d": VGG1D, "alexnet1d": Alex1D, "rnn_lstm": AudioRNN}


@st.cache_resource
def load_m2(key):
    c = ckpt(f"m2_{key}.pth")
    m = AUDIO_CLS[key](len(c["classes"]))
    m.load_state_dict(c["state"])
    return m.eval(), c


def mfcc_segments(y, sr=16000, seg_s=3, nseg=6):
    import librosa
    L = seg_s * sr
    out = []
    for s in list(range(0, max(len(y) - L + 1, 1), L))[:nseg]:
        seg = y[s:s + L]
        if len(seg) < L:
            seg = np.pad(seg, (0, L - len(seg)))
        out.append(librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=40, hop_length=512).astype(np.float32))
    return np.stack(out)


# ---------------------------------------------------------------- Module 1 (video)
class Head(nn.Module):
    def __init__(s, d, kind):
        super().__init__()
        s.r = (nn.LSTM if kind == "lstm" else nn.RNN)(d, 128, batch_first=True)
        s.drop = nn.Dropout(0.3)
        s.o = nn.Linear(128, 7)

    def forward(s, x):
        o, _ = s.r(x)
        return s.o(s.drop(o[:, -1]))


@st.cache_resource
def load_m1(key):
    name, kind = key.rsplit("_", 1)
    c = ckpt(f"m1_{key}.pth")
    feats = models.vgg16(weights=None).features if name.startswith("vgg16") else models.alexnet(weights=None).features
    feats.load_state_dict(ckpt(f"m1_{name}_backbone.pth"))
    head = Head(c["d"], kind)
    head.load_state_dict(c["state"])
    return feats.eval(), head.eval(), c["mu"], c["sd"]


def video_windows(path, T=8, step=48, max_windows=12):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < T:
        return [], []
    starts = list(range(0, max(n - step + 1, 1), step))
    if len(starts) > max_windows:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, max_windows).astype(int)]
    wins, motion = [], []
    for s in starts:
        idx = np.linspace(s, min(s + step - 1, n - 1), T).astype(int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, im = cap.read()
            if not ok:
                frames = None
                break
            frames.append(im)
        if not frames:
            continue
        gray = [cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
        motion.append(float(np.mean([np.abs(gray[i + 1] - gray[i]).mean() for i in range(T - 1)])))
        wins.append(np.stack([cv2.cvtColor(cv2.resize(f, (224, 224)), cv2.COLOR_BGR2RGB) for f in frames]))
    cap.release()
    return wins, motion


tab1, tab2, tab3, tab4 = st.tabs(["Visual Triage & Species Scanner", "Acoustic Cough & Distress Listener",
                                  "Video Behavior & Activity Tracker", "Syllabus Model Benchmarks"])

# ================================================================ TAB 1
with tab1:
    st.subheader("Droppings pathology screening (VGG16 / AlexNet + Grad-CAM)")
    r3 = load_json("module3_results.json")
    if not r3:
        st.error(f"module3_results.json not found in {OUT}")
    else:
        classes = r3["classes"]
        key = st.selectbox("Model", ["vgg16_transfer", "alexnet_transfer", "vgg16_scratch", "alexnet_scratch"], key="m3sel")
        up = st.file_uploader("Upload a poultry droppings image", type=["jpg", "jpeg", "png"], key="img")
        if up:
            img = Image.open(up).convert("RGB")
            x = IMG_TF(img).unsqueeze(0)
            m = load_m3(key, len(classes))
            with torch.no_grad():
                p = torch.softmax(m(x), 1)[0].numpy()
            c1, c2, c3 = st.columns(3)
            c1.image(img, caption="Input", use_container_width=True)
            top = classes[int(p.argmax())]
            c2.metric("Prediction", top, f"{p.max() * 100:.1f}% confidence")
            c2.bar_chart(pd.DataFrame({"probability": p}, index=classes))
            try:
                from pytorch_grad_cam import GradCAM
                from pytorch_grad_cam.utils.image import show_cam_on_image
                for q in m.parameters():
                    q.requires_grad = True
                cam = GradCAM(model=m, target_layers=[m.features[-1]])
                g = cam(input_tensor=x, targets=None)[0]
                base = np.asarray(img.resize((224, 224))).astype(np.float32) / 255
                c3.image(show_cam_on_image(base, g, use_rgb=True), caption="Grad-CAM heatmap", use_container_width=True)
            except Exception as e:
                gp = os.path.join(OUT, "gradcam_demo.png")
                if os.path.exists(gp):
                    c3.image(gp, caption="Saved Grad-CAM examples (live Grad-CAM unavailable)", use_container_width=True)
                c3.caption(f"Live Grad-CAM unavailable: {e}")
            st.caption("Heatmaps are indicative of where the model looked, not proof of disease location.")
        else:
            gp = os.path.join(OUT, "gradcam_demo.png")
            if os.path.exists(gp):
                st.image(gp, caption="Example Grad-CAM output from training", use_container_width=True)

# ================================================================ TAB 2
with tab2:
    st.subheader("Vocalization monitor (1D VGG16 / 1D AlexNet / LSTM on MFCC)")
    if not os.path.exists(os.path.join(OUT, "m2_vgg1d.pth")):
        st.warning("Audio model files (m2_*.pth) not found in the outputs folder yet.")
    else:
        key = st.selectbox("Model", ["vgg1d", "alexnet1d", "rnn_lstm"], key="m2sel")
        up = st.file_uploader("Upload a .wav farm audio clip", type=["wav"], key="wav")
        if up:
            import librosa, librosa.display
            y, sr = librosa.load(up, sr=16000, duration=18)
            m, c = load_m2(key)
            X = (mfcc_segments(y) - c["mu"]) / c["sd"]
            with torch.no_grad():
                P = torch.softmax(m(torch.tensor(X, dtype=torch.float32)), 1).mean(0).numpy()
            names = c["classes"]
            pred = names[int(P.argmax())]
            alert = "unhealthy" in pred.lower()
            (st.error if alert else st.success)(f"{'RESPIRATORY ALERT' if alert else 'OK'}: predicted '{pred}' ({P.max() * 100:.1f}%)")
            fig, ax = plt.subplots(1, 2, figsize=(12, 3))
            ax[0].plot(np.arange(len(y)) / sr, y)
            ax[0].set_title("Waveform")
            ax[0].set_xlabel("s")
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
            ax[1].imshow(S, origin="lower", aspect="auto")
            ax[1].set_title("Mel spectrogram")
            st.pyplot(fig)
            st.bar_chart(pd.DataFrame({"probability": P}, index=names))

# ================================================================ TAB 3
with tab3:
    st.subheader("Behavior and activity tracker (frozen CNN features + LSTM/RNN)")
    if not os.path.exists(os.path.join(OUT, "m1_vgg16_transfer_lstm.pth")):
        st.warning("Video model files (m1_*.pth) not found in the outputs folder.")
    else:
        key = st.selectbox("Model", ["vgg16_transfer_lstm", "vgg16_transfer_rnn", "alexnet_transfer_lstm",
                                     "alexnet_transfer_rnn", "alexnet_scratch_lstm", "alexnet_scratch_rnn"], key="m1sel")
        thr = st.slider("Low-activity threshold (mean frame change)", 0.5, 15.0, 3.0, 0.5)
        up = st.file_uploader("Upload a short bird video", type=["mp4", "mov", "avi"], key="vid")
        if up:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(up.read())
            tmp.close()
            wins, motion = video_windows(tmp.name)
            if not wins:
                st.error("Could not read enough frames from this video.")
            else:
                feats, head, mu, sd = load_m1(key)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                rows = []
                with torch.no_grad():
                    for i, w in enumerate(wins):
                        x = (torch.from_numpy(w).permute(0, 3, 1, 2).float() / 255 - mean) / std
                        F = (feats(x).mean((2, 3)) - mu) / sd
                        p = torch.softmax(head(F.unsqueeze(0)), 1)[0]
                        rows.append({"window": i + 1, "behavior": BEH[int(p.argmax())], "confidence": round(float(p.max()), 3),
                                     "motion": round(motion[i], 2), "low activity": motion[i] < thr})
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)
                st.line_chart(df.set_index("window")["motion"])
                share = df["low activity"].mean()
                (st.error if share > 0.6 else st.success)(
                    f"{share * 100:.0f}% of windows are below the low-activity threshold "
                    f"{'- possible abnormal inactivity' if share > 0.6 else '- activity looks normal'}")
                st.caption("Models were trained on bird-cropped clips of 5 behaviors (Feeding, Preening, Swimming, Walking, Flying). "
                           "Here the full frame is used, so predictions are approximate. Inactivity is flagged by frame-difference motion, "
                           "not by the classifier.")

# ================================================================ TAB 4
with tab4:
    st.subheader("Benchmarks (held-out test sets)")
    rows = []
    for mod, fn in [("Module 1 video", "module1_results.json"), ("Module 2 audio", "module2_results.json"),
                    ("Module 3 images", "module3_results.json")]:
        r = load_json(fn)
        if not r:
            continue
        for k, v in r.items():
            if isinstance(v, dict) and "acc" in v:
                rows.append({"module": mod, "model": k, "accuracy": round(v["acc"], 3), "macro F1": round(v["f1"], 3)})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    pick = st.selectbox("Confusion matrix for", [f"{r['module']} | {r['model']}" for r in rows]) if rows else None
    if pick:
        mod, model = pick.split(" | ")
        r = load_json({"Module 1 video": "module1_results.json", "Module 2 audio": "module2_results.json",
                       "Module 3 images": "module3_results.json"}[mod])
        labels = r.get("behaviors") or r.get("classes")
        cm = np.array(r[model]["cm"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(cm)))
        ax.set_yticks(range(len(cm)))
        ax.set_xticklabels(labels[:len(cm)], rotation=45, ha="right")
        ax.set_yticklabels(labels[:len(cm)])
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)