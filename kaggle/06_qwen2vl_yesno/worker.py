
import sys, os, json
import numpy as np, torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

root, items_file, out_file, device = sys.argv[1:5]
M = "Qwen/Qwen2-VL-2B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(M, torch_dtype=torch.float32, attn_implementation="eager").to(device).eval()
proc = AutoProcessor.from_pretrained(M, min_pixels=224 * 392, max_pixels=224 * 392)   # 384x224 frames -> 392x224 -> 112 visual tokens each
tok = proc.tokenizer
yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["Yes", "yes", " Yes"]})
no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["No", "no", " No"]})

QUESTION = ("These are consecutive frames from a car dashcam, in time order. Does a traffic collision happen in these frames - "
            "the camera car or another vehicle hitting a vehicle, person or object? Answer Yes or No.")
WINDOWS = [np.linspace(0, 29, 8), np.linspace(0, 15, 8), np.linspace(14, 29, 8)]   # whole clip, first half, second half

def lse(x):
    return torch.logsumexp(x, dim=0)

done = set()
if os.path.exists(out_file):
    for line in open(out_file):
        r = json.loads(line); done.add((r["split"], r["clip_id"]))
items = [tuple(x) for x in json.load(open(items_file))]
with open(out_file, "a") as f:
    for k, (split, cid) in enumerate(items):
        if (split, cid) in done:
            continue
        frames = [Image.open(f"{root}/{split}/{cid}/frame_{j:03d}.jpg").convert("RGB") for j in range(30)]
        scores = []
        for w in WINDOWS:
            imgs = [frames[int(round(i))] for i in w]
            msg = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": QUESTION}]}]
            text = proc.apply_chat_template(msg, add_generation_prompt=True)
            inputs = proc(text=[text], images=imgs, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1].float()
            scores.append(float(lse(logits[yes_ids]) - lse(logits[no_ids])))
        f.write(json.dumps({"split": split, "clip_id": cid, "scores": scores}) + "\n"); f.flush()
        if k % 50 == 0:
            print(k, len(items), scores, flush=True)
