# Data notes

Raw working notes on dataset locations, folder layouts, and label
conventions — kept as-is (not prose) since they're the fastest reference
when writing a new loader. See [README.md](README.md) for the project
overview.

## ROSE-1
All 3mm
- Path on Oscar: /files22_lrsresearch/ENG_Lee-Lab_Shared/group/data/publi
- 117 images from 39 subjects (26 Alzheimer's, 13 healthy)
- 3x3mm FOV, 304x304px, .tif format
- Split: 30 train / 9 test per layer (ignoring this, will use k-fold)
- Three layers: SVC, DVC, SVC_DVC
- Use img/ for images, gt/ for ground truth masks
- SVC: ['gt', 'img', 'thick_gt', 'thick_gt_converted', 'thin_gt']
- DVC: ['.DS_Store', 'gt', 'img']
- SVC_DVC: ['gt', 'img']


## OCTA-500
- Path: /files22_lrsresearch/ENG_Lee-Lab_Shared/group/data/public/OCTA_500
- 200 subjects (IDs 10301-10500), 3x3mm FOV, 304x304px, .bmp format
- Same resolution as ROSE-1 — good for cross-dataset generalization testing
- Images: OCTA_3mm_part3/OCTA_3mm/Projection Maps/
- Labels: Label/GT_LargeVessel/ (500 total, use 10301-10500 for 3mm)
- OCTA-500 -> ROSE-1: ILM_OPL projection -> SVC, OPL_BM -> DVC, full -> SVC_DVC