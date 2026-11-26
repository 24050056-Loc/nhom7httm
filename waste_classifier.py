"""
Phân loại: LON NƯỚC / CỤC GIẤY / NẮP CHAI NHỰA
Mô hình: EfficientNet-B0 (pretrained ImageNet) + transfer learning.

Cài thư viện:
    pip install torch torchvision pillow

Cách dùng:
    1) python waste_classifier.py prepare --src .        # xếp ảnh vào dataset/ theo tên file
       (lonnuoc*.jpg -> can, cucgiay*.jpg -> paper, napchai*.jpg -> cap)
    2) python waste_classifier.py train                  # train lần đầu
    3) python waste_classifier.py predict                # mở hộp thoại chọn ảnh để nhận diện

Train thêm sau này (khi có thêm ảnh mới bỏ vào dataset/<lớp>/):
    python waste_classifier.py train --resume            # học tiếp từ model cũ
    python waste_classifier.py train --resume --unfreeze # fine-tune sâu hơn (khi có >= ~30 ảnh/lớp)

Cấu trúc dữ liệu:
    dataset/
        can/     <- lon nước
        paper/   <- cục giấy
        cap/     <- nắp chai nhựa
"""
import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms

DATA_DIR = Path("dataset")
MODEL_PATH = Path("waste_model.pth")
IMG_SIZE = 224
CONF_THRESHOLD = 0.60  # dưới ngưỡng này sẽ báo "không chắc chắn"
LABEL_VI = {"can": "Lon nước", "paper": "Cục giấy", "cap": "Nắp chai nhựa"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_image(path):
    """Mở ảnh, ảnh PNG trong suốt sẽ được đặt lên nền trắng."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


class SampleDataset(Dataset):
    """Dataset có thể lặp lại mẫu nhiều lần (mỗi lần augment khác nhau) vì dữ liệu rất ít."""

    def __init__(self, samples, transform, repeat=1):
        self.samples, self.transform, self.repeat = samples, transform, repeat

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, i):
        path, label = self.samples[i % len(self.samples)]
        return self.transform(load_image(path)), label


def build_model(num_classes, pretrained=True):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ----------------------------------------------------------------- PREPARE
def prepare(src):
    """Xếp 6 ảnh ban đầu vào dataset/ theo tên file."""
    src = Path(src)
    for d in ("can", "paper", "cap"):
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)
    # tiền tố tên file -> thư mục lớp
    rules = {
        "lonnuoc": "can", "lon": "can",
        "cucgiay": "paper", "giay": "paper",
        "napchai": "cap", "win_": "cap",
    }
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    n = 0
    for f in sorted(src.iterdir()):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        name = f.name.lower()
        dst = next((cls for prefix, cls in rules.items() if name.startswith(prefix)), None)
        if dst is None:
            print(f"  Bỏ qua (không khớp quy tắc): {f.name}")
            continue
        shutil.copy(f, DATA_DIR / dst / f.name)
        print(f"  {f.name} -> {dst}")
        n += 1
    print(f"Đã chép {n} ảnh vào {DATA_DIR}/")


# ------------------------------------------------------------------- TRAIN
def train(epochs, lr, resume, unfreeze, batch_size):
    base = datasets.ImageFolder(DATA_DIR, loader=load_image)
    classes = base.classes
    samples = base.samples
    print(f"Lớp: {classes} | Tổng số ảnh: {len(samples)} | Thiết bị: {DEVICE}")
    for i, c in enumerate(classes):
        print(f"  {c}: {sum(1 for _, l in samples if l == i)} ảnh")

    model = build_model(len(classes))
    if resume and MODEL_PATH.exists():
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        if ckpt["classes"] != classes:
            raise SystemExit(f"Lớp không khớp model cũ: {ckpt['classes']} != {classes}")
        model.load_state_dict(ckpt["state_dict"])
        lr = min(lr, 3e-4)
        print("Đã nạp model cũ, train tiếp.")
    model.to(DEVICE)

    # Đóng băng backbone (tránh overfit khi ít ảnh); --unfreeze mở các block cuối
    for p in model.features.parameters():
        p.requires_grad = False
    if unfreeze:
        for p in model.features[5:].parameters():
            p.requires_grad = True
        lr = min(lr, 1e-4)

    # Chia validation chỉ khi đủ ảnh
    if len(samples) >= 30:
        n_val = max(len(classes), int(0.2 * len(samples)))
        tr_idx, va_idx = random_split(range(len(samples)), [len(samples) - n_val, n_val])
        tr_samples = [samples[i] for i in tr_idx]
        va_samples = [samples[i] for i in va_idx]
        val_loader = DataLoader(SampleDataset(va_samples, eval_tf), batch_size=batch_size)
    else:
        tr_samples, val_loader = samples, None
        print("Ít ảnh (<30): train trên toàn bộ, không có tập validation.")

    repeat = max(1, 300 // len(tr_samples))
    loader = DataLoader(SampleDataset(tr_samples, train_tf, repeat), batch_size=batch_size,
                        shuffle=True, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        if not unfreeze:
            model.features.eval()  # giữ BatchNorm của backbone ổn định
        tot_loss = correct = total = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        sched.step()
        msg = f"Epoch {ep:02d}/{epochs} loss={tot_loss/total:.4f} acc={correct/total:.3f}"

        score = correct / total
        if val_loader is not None:
            model.eval()
            vc = vt = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    vc += (model(x).argmax(1) == y).sum().item()
                    vt += len(y)
            score = vc / vt
            msg += f" val_acc={score:.3f}"
        print(msg)

        if score >= best:
            best = score
            torch.save({"state_dict": model.state_dict(), "classes": classes}, MODEL_PATH)

    print(f"Đã lưu model tốt nhất vào {MODEL_PATH} (score={best:.3f})")


# ----------------------------------------------------------------- PREDICT
def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit("Chưa có waste_model.pth - hãy chạy 'train' trước.")
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = build_model(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(DEVICE).eval(), ckpt["classes"]


@torch.no_grad()
def classify(model, classes, path):
    x = eval_tf(load_image(path)).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1)[0]
    conf, idx = probs.max(0)
    name = classes[idx.item()]
    label = LABEL_VI.get(name, name)
    if conf.item() < CONF_THRESHOLD:
        label = f"Không chắc chắn ({label}?)"
    return label, conf.item()


def predict_gui(files=None):
    import tkinter as tk
    from tkinter import filedialog
    from PIL import ImageTk

    model, classes = load_model()
    root = tk.Tk()
    root.title("Nhận diện: lon nước / cục giấy / nắp chai nhựa")

    if not files:
        files = filedialog.askopenfilenames(
            title="Chọn ảnh cần nhận diện",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp")])
    if not files:
        root.destroy()
        return

    thumbs = []
    cols = 3
    for i, f in enumerate(files):
        label, conf = classify(model, classes, f)
        print(f"{Path(f).name}: {label} ({conf*100:.1f}%)")
        img = load_image(f)
        img.thumbnail((260, 260))
        tk_img = ImageTk.PhotoImage(img)
        thumbs.append(tk_img)
        frame = tk.Frame(root, bd=1, relief="solid", padx=6, pady=6)
        frame.grid(row=i // cols, column=i % cols, padx=6, pady=6)
        tk.Label(frame, image=tk_img).pack()
        tk.Label(frame, text=Path(f).name, font=("Arial", 9)).pack()
        tk.Label(frame, text=f"{label}\n{conf*100:.1f}%",
                 font=("Arial", 13, "bold"), fg="#0a7d2c").pack()

    root.mainloop()


# -------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("prepare", help="xếp ảnh ban đầu vào dataset/")
    p.add_argument("--src", default=".")

    t = sub.add_parser("train", help="huấn luyện / huấn luyện tiếp")
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--resume", action="store_true", help="train tiếp từ waste_model.pth")
    t.add_argument("--unfreeze", action="store_true", help="fine-tune các block cuối của backbone")

    r = sub.add_parser("predict", help="chọn ảnh để nhận diện")
    r.add_argument("files", nargs="*", help="(tuỳ chọn) đường dẫn ảnh, bỏ trống để mở hộp thoại chọn")

    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.src)
    elif args.cmd == "train":
        train(args.epochs, args.lr, args.resume, args.unfreeze, args.batch)
    else:
        predict_gui(getattr(args, "files", None))


if __name__ == "__main__":
    main()
