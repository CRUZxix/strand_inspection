"""Live-video inspection engine (Phase 2).

Frame in -> structured status out. Three pieces:

  frame_sharpness   Laplacian-variance quality gate ("Hold steady" on motion blur)
  PacketTracker     full localisation only when the packet appears/moves; otherwise the
                    10 boxes are carried along by a cheap low-res belt-block track
  Inspector         state machine  SEARCHING -> LOCKED -> INSPECTING -> LATCHED
                    with multi-frame voting, so one glare/wrinkle frame can't flip a verdict

Boxes use the same 0-1000 normalised convention as src/localize.py.
"""
import time
from collections import Counter, deque

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.dataset import GOOD_CLASS
from src.infer import H, W, make_batch
from src.localize import (belt_mask, box_to_pixels, crop_fixed_aspect, grid_components,
                          solid_background_present)

SEARCHING, LOCKED, INSPECTING, LATCHED = "SEARCHING", "LOCKED", "INSPECTING", "LATCHED"


# ---------------------------------------------------------------- quality gate
def frame_sharpness(img, width=640):
    """Variance of the Laplacian on a fixed-width grayscale copy (resolution independent)."""
    g = np.asarray(img.convert("L"))
    if g.shape[1] != width:
        g = cv2.resize(g, (width, max(1, round(g.shape[0] * width / g.shape[1]))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ---------------------------------------------------------------- tracking
def _block_bbox(img, table):
    """Bounding box (0-1000) of the dominant belt pixels on a ~256px mask, or None."""
    m = belt_mask(img, max_side=256, reject_solid=table)
    if m.mean() < 0.01:
        return None
    ys, xs = np.nonzero(m)
    h, w = m.shape
    x0, x1 = np.percentile(xs, [1, 99]); y0, y1 = np.percentile(ys, [1, 99])
    return np.array([1000 * x0 / w, 1000 * y0 / h, 1000 * x1 / w, 1000 * y1 / h])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _warp(box, ref, cur):
    """Map a component box from the reference block frame to the current block frame."""
    sx = (cur[2] - cur[0]) / max(ref[2] - ref[0], 1e-6); sy = (cur[3] - cur[1]) / max(ref[3] - ref[1], 1e-6)
    return [cur[0] + (box[0] - ref[0]) * sx, cur[1] + (box[1] - ref[1]) * sy,
            cur[0] + (box[2] - ref[0]) * sx, cur[1] + (box[3] - ref[1]) * sy]


class PacketTracker:
    """Full localisation (~400 ms) only while acquiring a packet; ~10 ms block tracking after.

    Full localisation is itself jittery under hand shake (box IoU between consecutive frames
    can drop to ~0.2), so a lock is built from `lock_samples` full localisations whose boxes
    are warped into one frame and median-combined. Afterwards the locked boxes are carried
    along by the belt-block bounding box; a new acquisition starts only if the block overlap
    with the lock falls below `track_iou` (packet moved/replaced)."""

    def __init__(self, track_iou=0.65, same_iou=0.85, lock_samples=5):
        self.track_iou, self.same_iou, self.lock_samples = track_iou, same_iou, lock_samples
        self.reset()

    def reset(self):
        self.ref = None; self.comps = None; self.table = False; self.samples = []

    @property
    def locked(self):
        return self.comps is not None

    def update(self, img):
        """Returns (components | None, info). info: relocalized, iou, stable, jump."""
        none = {"relocalized": False, "iou": 0.0, "stable": False, "jump": False}
        bb = _block_bbox(img, self.table)
        if bb is None:
            return None, none
        i = iou(bb, self.ref) if self.ref is not None else 0.0
        if self.locked and i >= self.track_iou:
            comps = [{**c, "bbox": _warp(c["bbox"], self.ref, bb)} for c in self.comps]
            return comps, {"relocalized": False, "iou": i, "stable": i >= self.same_iou, "jump": False}
        jump = self.locked                      # we had a lock and lost it -> packet moved/replaced
        if jump:
            self.comps = None; self.samples = []
        try:                                    # acquire: one more full localisation sample
            if not self.samples:                # table check once per acquisition (it is the slow part)
                self.table = solid_background_present(img)
            comps, _ = grid_components(img, 2, 5, 0.72, split="mid", col_overlap=0.08, reject_solid=self.table)
        except ValueError:
            self.reset()
            return None, none
        bb = _block_bbox(img, self.table)
        if bb is None:
            self.reset(); return None, none
        if self.samples and iou(bb, self.samples[0][1]) < 0.7:   # packet moving while sampling
            self.samples = []
        self.samples.append((comps, bb))
        info = {"relocalized": True, "iou": i, "stable": False, "jump": jump}
        if len(self.samples) < self.lock_samples:
            return comps, info                  # provisional boxes while acquiring
        self.ref = bb
        merged = []
        for k, c in enumerate(comps):
            boxes = np.array([_warp(sc[k]["bbox"], sb, bb) for sc, sb in self.samples])
            merged.append({**c, "bbox": np.median(boxes, axis=0).tolist()})
        self.comps, self.samples = merged, []
        return merged, info


# ---------------------------------------------------------------- voting
def consensus(votes, min_share=0.8):
    """votes: list of per-frame (counts[10], conf[10]). Returns (counts, conf, settled) with the
    per-component majority count; settled only if EVERY component's majority share >= min_share."""
    n = len(votes)
    counts, confs, settled = [], [], True
    for k in range(len(votes[0][0])):
        c = Counter(v[0][k] for v in votes)
        top, num = c.most_common(1)[0]
        counts.append(top)
        confs.append(float(np.mean([v[1][k] for v in votes if v[0][k] == top])))
        if num / n < min_share:
            settled = False
    return counts, confs, settled


class Inspector:
    """Stateful per-stream inspector. Call process(frame) for every incoming frame."""

    def __init__(self, classifier, conf_thr=0.6, blur_thr=40.0, window=5, min_votes=3,
                 give_up=15, lost_frames=5, stable_frames=2):
        self.clf, self.conf_thr, self.blur_thr = classifier, conf_thr, blur_thr
        self.window, self.min_votes, self.give_up = window, min_votes, give_up
        self.lost_frames, self.stable_frames = lost_frames, stable_frames
        self.tracker = PacketTracker()
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.state, self.verdict, self.votes = SEARCHING, None, deque(maxlen=self.window)
        self.lost = self.stable = self.inspected = 0
        self.last_counts = self.last_confs = self.last_comps = None

    # -- helpers
    def _boxes(self, img, comps, counts=None, confs=None, final=False):
        out = []
        for k, c in enumerate(comps):
            b = {"id": c["position"], "coords": list(box_to_pixels(c["bbox"], img.width, img.height))}
            if counts is not None:
                b["count"] = int(counts[k]); b["conf"] = round(float(confs[k]), 3)
                if confs[k] < self.conf_thr:
                    b["status"] = "UNCERTAIN"
                else:
                    b["status"] = "GOOD" if counts[k] == GOOD_CLASS else "DEFECT"
                if not final:
                    b["status"] = "ALIGNING" if b["status"] == "UNCERTAIN" else b["status"]
            else:
                b["status"] = "ALIGNING"
            out.append(b)
        return out

    def _payload(self, img, t0, hint="", boxes=None, sharp=None, relocalized=False, blur=None):
        return {"state": self.state, "packet_verdict": self.verdict, "hint": hint,
                "boxes": boxes if boxes is not None else [], "blur_var": None if blur is None else round(blur, 1),
                "relocalized": relocalized, "votes": len(self.votes),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}

    def _boxes_cached(self, img):
        if self.last_comps is None:
            return []
        return self._boxes(img, self.last_comps, self.last_counts, self.last_confs, final=self.state == LATCHED)

    # -- main entry
    def process(self, image):
        t0 = time.perf_counter()
        img = ImageOps.exif_transpose(image).convert("RGB")
        blur = frame_sharpness(img)
        if blur < self.blur_thr:   # motion blur: never consume a vote or change the verdict...
            if _block_bbox(img, self.tracker.table) is None:   # ...but an EMPTY view is not blur
                self.lost += 1
                if self.lost >= self.lost_frames and self.state != SEARCHING:
                    self.reset()
                return self._payload(img, t0, "Point camera at packet",
                                     self._boxes_cached(img) if self.state != SEARCHING else [], blur=blur)
            self.lost = 0
            return self._payload(img, t0, "Verdict latched" if self.state == LATCHED else "Aligning - hold steady",
                                 self._boxes_cached(img), blur=blur)

        comps, info = self.tracker.update(img)
        if comps is None:
            self.lost += 1
            if self.lost >= self.lost_frames and self.state != SEARCHING:
                self.reset()
            return self._payload(img, t0, "Point camera at packet",
                                 self._boxes_cached(img) if self.state != SEARCHING else [], blur=blur)
        self.lost = 0

        if info["jump"] and self.state in (INSPECTING, LATCHED):
            # packet jumped / swapped: old votes and verdict no longer describe what's in view
            self.state, self.verdict, self.votes, self.inspected = LOCKED, None, deque(maxlen=self.window), 0
            self.last_counts = self.last_confs = None
            self.stable = 0
        self.last_comps = comps

        if self.state == SEARCHING:
            self.state, self.stable = LOCKED, 0

        if self.state == LOCKED:
            self.stable = self.stable + 1 if info["stable"] and self.tracker.locked else 0
            if self.stable >= self.stable_frames:
                self.state = INSPECTING
            return self._payload(img, t0, "Locked - hold steady", self._boxes(img, comps),
                                 relocalized=info["relocalized"], blur=blur)

        if self.state == LATCHED:    # verdict stays until packet leaves or jumps
            return self._payload(img, t0, "Verdict latched", self._boxes_cached(img),
                                 relocalized=info["relocalized"], blur=blur)

        # INSPECTING: one batched pass, add a vote
        crops = [crop_fixed_aspect(img, box_to_pixels(c["bbox"], img.width, img.height), W / H, 0.04)
                 .resize((W, H), Image.LANCZOS) for c in comps]
        prob = self.clf.probs(make_batch(crops))
        counts, confs = prob.argmax(1).tolist(), prob.max(1).tolist()
        self.votes.append((counts, confs)); self.inspected += 1
        if len(self.votes) >= self.min_votes:
            vc, vf, settled = consensus(list(self.votes))
            self.last_counts, self.last_confs = vc, vf
            if settled:
                weak = any(f < self.conf_thr for f in vf)
                if weak:
                    self.verdict = "UNCERTAIN"
                else:
                    self.verdict = "PASS" if all(c == GOOD_CLASS for c in vc) else "FAIL"
                self.state = LATCHED
                return self._payload(img, t0, "Verdict latched", self._boxes_cached(img),
                                     relocalized=info["relocalized"], blur=blur)
            if self.inspected >= self.give_up:
                self.verdict, self.state = "UNCERTAIN", LATCHED
                return self._payload(img, t0, "Unstable - re-scan packet", self._boxes_cached(img), blur=blur)
        else:
            self.last_counts, self.last_confs = counts, confs
        return self._payload(img, t0, f"Inspecting {len(self.votes)}/{self.min_votes}+",
                             self._boxes_cached(img), relocalized=info["relocalized"], blur=blur)
