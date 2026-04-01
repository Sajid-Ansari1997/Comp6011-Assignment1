# COMP6011 Task 1 — Rubric-Aligned Code Pack

This code pack is designed to support the rubric items for:
- compulsory additional materials
- proof-of-concept evidence
- benchmarking candidates
- minimum two datasets
- quantitative and qualitative results
- deployment awareness
- carbon footprint estimation
- learning evidence through logs and experiment tracking

## Included capabilities

1. Train and evaluate multiple candidate models
2. Support at least two datasets:
   - BDD100K
   - Cityscapes
3. Save quantitative metrics
4. Save qualitative prediction images
5. Generate model trade-off plots
6. Export deployment-ready ONNX models
7. Estimate carbon emissions
8. Save experiment metadata and logs

## Folder structure

- `code/` scripts
- `configs/` dataset and experiment configs
- `notes/` learning process templates
- `outputs/` generated results

## Recommended workflow

### 1. Install packages
```bash
pip install -r requirements.txt
```

### 2. Train one or more candidate models
```bash
python code/train.py --config configs/experiment_bdd.yaml
python code/train.py --config configs/experiment_cityscapes.yaml
```

### 3. Benchmark trained models
```bash
python code/benchmark.py --config configs/experiment_bdd.yaml
python code/benchmark.py --config configs/experiment_cityscapes.yaml
```

### 4. Save qualitative results
```bash
python code/qualitative_results.py --weights outputs/yolov8n_bdd/weights/best.pt --source /path/to/sample/images --outdir outputs/qualitative/yolov8n_bdd
```

### 5. Export ONNX for deployment discussion
```bash
python code/export_model.py --weights outputs/yolov8n_bdd/weights/best.pt --format onnx
```

### 6. Plot cross-model comparison
```bash
python code/plot_benchmarks.py --metrics_dir outputs/metrics
```

### 7. Estimate carbon
```bash
python code/carbon_estimate.py --config configs/experiment_bdd.yaml
```

## Important

You must edit dataset paths in the config files before running.
The code supports the rubric, but your report must still explain:
- why each candidate was chosen
- how metrics are interpreted
- why the final model fits freight-truck perception
- ethical, privacy, and deployment implications
