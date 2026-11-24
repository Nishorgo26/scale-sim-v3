# Accuracy

Accuracy testing for ResNet-18 using baseline, PoT, APoT, and Mitchell multiplication.

Run all commands from the repository root using Python 3.11.

## Install

Install the required Python packages:

```bash
python -m pip install -r project/accuracy/requirements.txt
```

## Test

Run the unit tests:

```bash
python -m unittest project.accuracy.shiftmac_tests -v
```

Show available accuracy options:

```bash
python -m project.accuracy.run --help
```

## Dataset Setup

The accuracy experiments use the Imagenette validation dataset, which contains 10 classes from ImageNet.

Download and extract Imagenette:

```bash
mkdir -p datasets
cd datasets

curl -L -o imagenette2-160.tgz \
  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz

tar -xzf imagenette2-160.tgz
cd ..
```

Create the class map to match Imagenette's folders with the original ImageNet-1K class indices:

```bash
python - <<'PY'
import json

classes = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701
}

with open("datasets/imagenette-class-map.json", "w") as f:
    json.dump(classes, f, indent=2)
PY
```

The dataset should now be located at:

```text
datasets/
├── imagenette2-160/
│   ├── train/
│   └── val/
└── imagenette-class-map.json
```

The dataset and pretrained model weights should not be committed to Git.

## Run Accuracy Experiments

The experiments use pretrained ResNet-18 weights (`IMAGENET1K_V1`) and seed 0.

### Baseline, PoT, and APoT

Run the 500-image comparison:

```bash
python -m project.accuracy.run \
  --data datasets/imagenette2-160/val \
  --class-map datasets/imagenette-class-map.json \
  --pes baseline pot apot \
  --max-images 500 \
  --batch-size 8 \
  --bits 4 \
  --seed 0 \
  --out project/results/accuracy_500.csv \
  --download-weights
```

### Mitchell

Run the 100-image comparison:

```bash
python -m project.accuracy.run \
  --data datasets/imagenette2-160/val \
  --class-map datasets/imagenette-class-map.json \
  --pes baseline mitchell \
  --max-images 100 \
  --batch-size 1 \
  --seed 0 \
  --out project/results/accuracy_mitchell_100.csv \
  --download-weights
```

The `--download-weights` option is only needed if the pretrained model weights are not already cached.

Mitchell is significantly slower because it approximates multiplication inside convolution and linear layers. Its implementation uses chunked operations to reduce memory usage and execution time.

Results are saved in `project/results/`.

---

# Accuracy Results

## Setup

We evaluated pretrained ResNet-18 using the Imagenette validation dataset.

PoT and APoT were evaluated on 500 images using `bits=4`. Mitchell was evaluated on 100 images due to its slower multiplication operations.

All experiments used seed 0 and the same pretrained ResNet-18 weights (`IMAGENET1K_V1`).

## Results

| PE | Images | Top-1 Accuracy | Difference from Baseline |
|---|---:|---:|---:|
| Baseline | 500 | 68.60% | Reference |
| PoT | 500 | 28.00% | -40.60 pp |
| APoT | 500 | 69.20% | +0.60 pp |
| Baseline | 100 | 63.00% | Reference |
| Mitchell | 100 | 64.00% | +1.00 pp |

## Findings

PoT caused a substantial reduction in classification accuracy. APoT maintained accuracy close to the original model, while Mitchell also produced similar accuracy on its smaller evaluation subset. Mitchell inference was slower because it approximates multiplication directly. An optimization reduced the 10-image experiment runtime from 15 minutes 12 seconds to 7 minutes 6 seconds. The 100-image Mitchell experiment took 68 minutes 23 seconds.

## Limitations

The experiments used Imagenette, a 10 class subset of ImageNet, rather than the full ImageNet validation dataset. Mitchell was evaluated on 100 images due to runtime limitations. Its accuracy should be compared against the matching 100 image baseline rather than the 500 image results. PoT and APoT approximate model weights using floating-point calculations, while Mitchell approximates products during inference. These experiments do not fully simulate fixed-point INT8 hardware arithmetic. Small accuracy improvements over baseline should not be interpreted as evidence that approximate multiplication generally improves classification accuracy.
