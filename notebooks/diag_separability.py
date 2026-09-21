"""CLAUDE.md §4 records that synthetic foreground is threshold-separable at IoU 0.9998,
which is WHY `occupancy: none` + image worked there. Does that hold on MRI?
If not, the decoder is blind to structure boundaries on this corpus."""
import sys, json, numpy as np
sys.path.insert(0, '.')
from src.data import Corpus, load_nifti
c = Corpus.load('data/mri')
scenes = list(dict.fromkeys(json.loads(l)['scene'] for l in open('data/mri/val.jsonl')))[:10]
best_all, per_class = [], {}
for s in scenes:
    img = load_nifti(c.root/'scenes'/s/'image.nii.gz', np.float32)
    lab = load_nifti(c.root/'scenes'/s/'labels.nii.gz', np.int16)
    img = (img - img.mean())/(img.std()+1e-8)          # data.normalize: zscore
    fg = lab > 0
    best = 0.0
    for t in np.percentile(img[img > img.min()], np.arange(50, 100, 1)):
        p = img > t
        iou = (p & fg).sum()/max((p | fg).sum(), 1)
        best = max(best, iou)
    best_all.append(best)
    for i, nm in enumerate(c.vocab.names, 1):
        m = lab == i
        if m.sum(): per_class.setdefault(nm, []).append((img[m].mean(), img[m].std()))
print(f"best achievable IoU for 'labels>0' by ANY global threshold, {len(scenes)} MRI scenes:")
print(f"   mean {np.mean(best_all):.4f}   (synthetic: 0.9998)")
print("\nper-class intensity (z-scored): can intensity even separate structures?")
bg = []
for s in scenes:
    img = load_nifti(c.root/'scenes'/s/'image.nii.gz', np.float32)
    lab = load_nifti(c.root/'scenes'/s/'labels.nii.gz', np.int16)
    img = (img - img.mean())/(img.std()+1e-8)
    bg.append(img[(lab == 0) & (img > img.min())].mean())
print(f"   non-structure tissue mean {np.mean(bg):+.3f}")
for nm in sorted(per_class, key=lambda n: np.mean([a for a,_ in per_class[n]]))[:6]:
    a = np.mean([x for x,_ in per_class[nm]]); sd = np.mean([y for _,y in per_class[nm]])
    print(f"   {nm:24s} {a:+.3f} +/- {sd:.3f}")
print("   ...")
for nm in sorted(per_class, key=lambda n: np.mean([a for a,_ in per_class[n]]))[-4:]:
    a = np.mean([x for x,_ in per_class[nm]]); sd = np.mean([y for _,y in per_class[nm]])
    print(f"   {nm:24s} {a:+.3f} +/- {sd:.3f}")
