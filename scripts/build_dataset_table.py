"""Build the RoboTwin 2.0 dataset table: task thumbnail, embodiment support, ACT score."""
import base64, io, json, os, sys
import cv2, numpy as np

R = "/coc/flash7/bli678/Projects/RoboTwin"
SCRATCH = "/tmp/claude-3582647/-coc-flash7-bli678-Projects-XPolicyLab/993275a1-dd44-456a-8fd3-8323e78492e9/scratchpad"
DD = "/coc/flash7/scratch/robotwin_shared/dataset"
EMB = [("aloha-agilex", "Aloha-AgileX"), ("arx-x5", "ARX-X5"),
       ("franka", "Franka"), ("piper", "Piper"), ("ur5", "UR5")]

lb = {r["task"]: r for r in json.load(open(f"{SCRATCH}/act_leaderboard.json"))}
tasks = sorted(os.listdir(DD))

def thumb(task, w=176):
    p = f"{R}/figs/tasks/{task}_home.png"
    if not os.path.exists(p):
        return None
    im = cv2.imread(p)
    h = int(im.shape[0] * w / im.shape[1])
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode() if ok else None

rows, n_img = [], 0
for t in tasks:
    have = set(os.listdir(os.path.join(DD, t)))
    sup = [e for e, _ in EMB if f"{e}_clean_50" in have]
    b64 = thumb(t)
    n_img += b64 is not None
    rows.append(dict(task=t, display=lb[t]["display"], sup=sup,
                     easy=lb[t]["act_easy"], hard=lb[t]["act_hard"], img=b64))
print(f"{len(rows)} tasks, {n_img} thumbnails embedded", file=sys.stderr)
json.dump(rows, open(f"{SCRATCH}/dataset_table.json", "w"))
