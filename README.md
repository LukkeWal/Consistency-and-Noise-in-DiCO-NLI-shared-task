# How to run

Run preprocessing to clean data and create a reproducible train/validation split:

```bash
python PreProcessing.py
```

Train and evaluate a model (`--save` saves it)<br>
See all options with `python Main.py --help`. 

```bash
python Main.py --model bert-base-uncased --save
```

Evaluate a saved model by result ID:

```bash
python Main.py --evaluate-saved 1
```

Training results, settings and metrics are saved in `results.csv`. Create plots by result ID:

```bash
python Visualization.py 1 loss
python Visualization.py 1 confusion-matrix
```

## Modules

- `PreProcessing.py`: cleans and splits the data.
- `DataHandler.py`: loads data and encodes labels.
- `EncoderTraining.py`: trains encoder models.
- `Evaluator.py`: evaluates models and records results.
- `Main.py`: connects the pipeline.
- `Visualization.py`: plots results.
