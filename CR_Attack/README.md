# A Certified Radius-Guided Attack Framework to Image Segmentation Models

Demo code for paper submission "A Certified Radius-Guided Attack Framework to Image Segmentation Models". More code will be released soon.

## Preparations
Download official torch model for Pascal VOC PSPNet [here](https://drive.google.com/drive/folders/1K18bS_WeUQH4O6qAZzCDoAfVg9VhgGMK).

Place it under ./modeldata and rename it to voc_psp_official.pth.

Download Pascal VOC Dataset with SBD augmentation with our automatic script.
```
cd VOC2012
chmod +x download.sh
./download.sh
```

## Whitebox Attack Examples
White-box CR-PGD Attack(with L2 norm=1, hyperameters all under default setting):
```
python3 whitebox_attack.py --config=configvoc_psp.json --norm=2 --eps=1
```
White-box CR-PGD Attack(with L-infty norm=0.004, hyperameters all under default setting):
```
python3 whitebox_attack.py --config=configvoc_psp.json --norm=inf --eps=0.004
```
# CISPA-WORK
