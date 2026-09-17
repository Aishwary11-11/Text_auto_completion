# Next Word Predictor with LSTM

A Python application that suggests the three most likely next words using a
TensorFlow/Keras LSTM. Type in the Tkinter window and click a suggestion to append
it, or request predictions from the terminal.

Train on the included Sherlock Holmes text or your own `.txt` / `.docx` corpus.
The saved `.keras` file contains both the model and its vocabulary, so later runs
do not need the original training file.

## Features

- Three clickable next-word suggestions with model probabilities.
- Debounced background inference so typing and pasting stay responsive.
- Custom TXT and DOCX training data, with case-insensitive tokenization.
- Automatic training on first run and model reuse afterward.
- Configurable paths and training settings without editing source code.
- Terminal prediction and training modes for use without a graphical display.

## Installation

Use 64-bit Python 3.11 (the version tested for this project). Download or clone
this repository, open a terminal in its directory, and create a virtual environment:

```sh
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of
`python` in the following commands. Activation is optional.

Tkinter is included with standard Python installations on Windows. Some Linux
distributions require their `python3-tk` system package. Check it with:

```sh
python -m tkinter
```

The default installation uses the CPU on native Windows. See the official
[TensorFlow installation guide](https://www.tensorflow.org/install/pip) for GPU
and platform requirements.

## Usage

### Launch the application

```sh
python word_predict.py
```

On the first run, the script trains on
`sherlock-holm.es_stories_plain-text_advs.txt`, saves `models/next_word.keras`, and
opens the GUI. Training happens before the window opens and can take a while on
a CPU; progress appears in the terminal. No pretrained weights are bundled.

Subsequent runs load the saved model directly. The default paths are relative to
the script, so they also work when you launch it from another directory.
Explicit relative paths are resolved from your current working directory.

Type some words, wait for suggestions, and click one to append it. Clearing the
input clears the suggestions. Predictions from an earlier edit are discarded if
the text changes while inference is running.

### Train on your own corpus

```sh
python word_predict.py --text-file "data/my_corpus.txt" --model-path "models/custom.keras" --epochs 10 --train-only
```

For Word documents, pass a `.docx` file instead. Paragraph text is read; text in
tables, headers, and footers is not included. TXT files are decoded as UTF-8
(including BOM), Windows-1252, or Latin-1, in that order.

The corpus must contain at least `sequence-length + 1` whitespace-separated words.
With the default sequence length of 10, that means at least 11 words. A larger,
representative corpus is needed for useful predictions.

To replace an existing model with a newly trained one, explicitly use `--retrain`:

```sh
python word_predict.py --text-file "data/my_corpus.docx" --model-path "models/custom.keras" --retrain --train-only
```

Without `--retrain`, an existing model is reused. Corpus and architecture options
apply only when training; loading uses the saved vocabulary and sequence length.

### Predict in the terminal

```sh
python word_predict.py --predict "sherlock holmes"
python word_predict.py --model-path "models/custom.keras" --predict "once upon a"
```

These commands print up to three words and their model probabilities. They also
train first if the selected model does not exist. `--train-only` and `--predict`
are mutually exclusive and do not require a Tk window.

### Training options

| Option | Default | Purpose |
| --- | --- | --- |
| `--text-file` | Included Sherlock Holmes TXT | Training corpus |
| `--model-path` | `models/next_word.keras` | Model and vocabulary storage |
| `--sequence-length` | `10` | Maximum number of context words |
| `--epochs` | `10` | Training passes |
| `--batch-size` | `64` | Examples per training batch |
| `--embedding-dim` | `100` | Word embedding size |
| `--hidden-size` | `128` | Units in each LSTM layer |
| `--num-layers` | `3` | Number of stacked LSTM layers |
| `--seed` | `42` | Random seed used when training |
| `--retrain` | Off | Train again and replace the selected model |
| `--train-only` | Off | Prepare/load the model and exit |
| `--predict TEXT` | Off | Print predictions and exit |

Run `python word_predict.py --help` for the available arguments.

## How it works

1. `TextDataset` lowercases and splits the corpus on whitespace, builds a sorted
   vocabulary, and reserves index zero for padding.
2. Each fixed-length sequence is paired with the word immediately following it.
   Training uses integer arrays and sparse categorical cross-entropy.
3. `LSTMModel` passes the sequence through a masked embedding, stacked LSTM
   layers, and a softmax output layer. The final 10% of examples are used for
   validation when there are at least 10 examples; smaller datasets use all
   examples for training.
4. Keras saves the model configuration, vocabulary, and weights together. This
   preserves the word-to-index mapping even if the corpus changes or is removed.
5. `NextWordPredictor` uses the last context words, ignores unknown words, pads
   shorter contexts, and ranks the output probabilities. Padding is never offered
   as a suggestion. Empty input or entirely unknown context returns no suggestions.
6. `NextWordPredictorGUI` waits 200 ms after an edit and runs inference in one
   background worker. All widget updates happen on the Tkinter thread.

This predicts the next complete word; it does not complete a partially typed
word. Punctuation remains attached to words (`holmes` and `holmes,` are distinct
tokens). Suggestions depend on the corpus and training, and probabilities are
model scores rather than a guarantee of correctness. The model has no internet
access and is not a pretrained language assistant.

Older model files created by the original script have no saved vocabulary. The
application asks you to retrain these with `--retrain --text-file ...` instead of
guessing their word mapping.

## Project structure

```text
word_predict.py                              Dataset, model, predictor, GUI, CLI
sherlock-holm.es_stories_plain-text_advs.txt   Included training corpus
requirements.txt                             Runtime dependencies
tests/test_word_predict.py                   Regression and integration tests
models/                                     Local saved models (not committed)
```

## Tests

```sh
python -m unittest discover -s tests -v
```

Tests cover TXT/DOCX loading, invalid corpora, next-word ranking, empty input,
unknown words, padding exclusion, model training and reload, command-line modes,
retraining, and GUI input/suggestion behavior. Small models are trained on
temporary corpora; the full bundled corpus is not trained during tests. Tkinter
tests skip when a display is unavailable.

## Contributing

Create a branch, make your changes, run the tests, and open a pull request.
Keep generated models, virtual environments, and personal corpora out of commits.

## License

The original README identifies the code as MIT-licensed, but the supplied project
does not include a `LICENSE` file. The included Arthur Conan Doyle corpus is a
separate work; retain any notices supplied with your training data.
