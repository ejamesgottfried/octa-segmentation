# Data

Raw data lives on Oscar. Paths belong in `src/paths.py` only. Raw data is never in git.
James's original working notes are in `docs/notes.md`.
Exception: `data/preprocessed/SVC/` (ROSE-1 after CLAHE + median) is tracked in git; see CLAUDE.md.

## ROSE-1 (public)

- Root: `/files22_lrsresearch/ENG_Lee-Lab_Shared/group/data/public/rose_dataset/ROSE-1`
- Device: Optovue RTVue XR Avanti, 3 x 3 mm, 304 x 304 px, `.tif`.
- SVC: official split, 30 train / 9 test (`SVC/{train,test}/{img,gt}`). Pixel-level manual GT.
  Vessel density about 23%. Also `thin_gt`, `thick_gt` variants (not yet used).
- DVC GT is centerline-only and parafoveal-ring-only: not usable for pixel Dice.
- `SVC_DVC` GT not characterized yet.
- AD/HC labels exist in the original paper but we do not have them.
- Preprocessed copy (CLAHE + median): `data/preprocessed/SVC/{train,test}/img` in this repo.
  Which CLAHE tile size made it (4 or 8) is unconfirmed; see CLAUDE.md known issues.
- OCTA-500 to ROSE-1 layer mapping: ILM_OPL to SVC, OPL_BM to DVC, FULL to SVC_DVC.

## OCTA-500 (public)

- Root: `/files22_lrsresearch/ENG_Lee-Lab_Shared/group/data/public/OCTA_500`
- Device: Optovue RTVue-XR, `.bmp`. No neurological labels (ophthalmic disease only).
- Input projection: `OCTA(ILM_OPL)` (paper projection B5). Both vessel labels were annotated on it.
- 3 mm subset: IDs 10301 to 10500, 304 x 304 px. Our split: `train_test_split(test_size=40,
  random_state=42)` on the sorted ID list (paper split differs; we do not use it).
- 6 mm subset: IDs 10001 to 10300, 400 x 400 px. Our split: `test_size=60, random_state=42`.
  Matches ARIAS field of view, so it is the training source for ARIAS transfer.
- Labels (`Label/Label/`):
  - `GT_LargeVessel`: manual, about 8% density.
  - `GT_Capillary`: **model-generated** (Li et al. Sec 3.5.3: network trained on 100 manual
    patches, sliding-window inference, FAZ-mask denoising, adversarial refinement, large
    vessels subtracted). About 48% density. Treat as pseudo-labels.
  - `GT_FAZ`: manual 2D. `GT_Artery`, `GT_Vein`, `GT_CAVF` also exist.
- Pixel spacing: 3 mm / 304 px ~ 9.9 um/px. Optical lateral resolution ~15 to 20 um, so
  single-capillary resolution is marginal even here.

## ARIAS (private, lab cohort)

- Owner: jlee123. Root path is in `src/paths.py`. Claude must never read it (see CLAUDE.md).
- Subject table for Claude: `private/arias_subjects_deid.csv` (gitignored), made by
  `scripts/deidentify_arias.py` from the raw sheet, which stays outside the repo.
  Columns: subject, group, sex, age, quality_left, quality_right, eye_left, eye_right,
  exclude, exclude_reason.
- Heidelberg Spectralis (a third vendor; absolute densities not comparable to Optovue/Zeiss
  literature). About 5.9 mm field at 11.55 um/px.
- 46 subjects (15 LR, 15 HR, 16 MCI), year 1, usually both eyes (~75 `.h5` files).
  Some subjects have missing eyes or corrupted/cross-contaminated files. Record those in
  the `exclude` columns of the de-identified table, not in tracked files.
  Effective N will be lower than 46.
- Volumes `octa`/`oct`: (512, 512, 496) float32, depth on axis 2. Layer volume `r1/lyr`
  uint8 codes 0 to 14. SVC = codes 2 to 6. DVC = code 8 (thin slab, projection artifact).
  Codes 2 to 8 (ILM to OPL) match OCTA-500's ILM_OPL projection.
- Useful per-file metadata: per-B-scan quality, SNR, scan eye, physical scale. DOB and visit
  date exist; only derived age may leave the extraction script.
- Capillaries are not resolvable at this sampling, so large-vessel segmentation is the target.
- No segmentation GT. Group labels come from jlee123.

## Cross-dataset notes

- Absolute Dice is not comparable across datasets (different vessel density floors,
  about 0.32 for ROSE and 0.64 for OCTA-500 capillaries).
- ROSE-1 model on ARIAS over-estimates vessel width (3 mm to 6 mm caliber mismatch).
