"""Train an LSTM and suggest the next word in a Tkinter window or the terminal."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


PAD_TOKEN = "<PAD>"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = PROJECT_DIR / "sherlock-holm.es_stories_plain-text_advs.txt"
DEFAULT_MODEL = PROJECT_DIR / "models" / "next_word.keras"


def tokenize(text):
    """Use identical, case-insensitive whitespace tokenization everywhere."""
    return text.lower().split()


class TextDataset:
    """Load a TXT/DOCX corpus and prepare fixed-length next-word examples."""

    def __init__(self, text_file_path, sequence_length=10):
        if sequence_length < 1:
            raise ValueError("Sequence length must be positive.")
        path = Path(text_file_path)
        if path.suffix.lower() == ".docx":
            text = self.convert_docx_to_text(path)
        elif path.suffix.lower() == ".txt":
            text = self.read_text_file(path)
        else:
            raise ValueError("Training data must be a .txt or .docx file.")

        self.sequence_length = sequence_length
        self.words = tokenize(text)
        if len(self.words) <= sequence_length:
            raise ValueError(
                f"The corpus needs at least {sequence_length + 1} words; "
                f"found {len(self.words)}. Use more text or a smaller --sequence-length."
            )
        self.vocabulary = [PAD_TOKEN, *sorted(set(self.words))]
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}
        self.index_to_word = dict(enumerate(self.vocabulary))

    @staticmethod
    def read_text_file(file_path):
        # Windows-1252 must precede Latin-1, which can decode every byte.
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                return Path(file_path).read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Cannot decode training file: {file_path}")

    @staticmethod
    def convert_docx_to_text(docx_path):
        from docx import Document

        document = Document(docx_path)
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    def prepare_training_data(self):
        indices = np.array([self.word_to_index[word] for word in self.words], dtype=np.int32)
        windows = np.lib.stride_tricks.sliding_window_view(indices, self.sequence_length + 1)
        return windows[:, :-1].copy(), windows[:, -1].copy()


@tf.keras.utils.register_keras_serializable(package="Custom", name="LSTMModel")
class LSTMModel(tf.keras.Model):
    """Serializable LSTM with the exact vocabulary used during training."""

    def __init__(self, vocab_size, embedding_dim=100, sequence_length=10,
                 hidden_size=128, num_layers=3, vocabulary=None, **kwargs):
        super().__init__(**kwargs)
        if min(vocab_size, embedding_dim, sequence_length, hidden_size, num_layers) < 1:
            raise ValueError("Model dimensions must be positive.")
        if vocabulary is not None and (
            len(vocabulary) != vocab_size
            or vocabulary[0] != PAD_TOKEN
            or any(not isinstance(word, str) or not word for word in vocabulary)
            or len(set(vocabulary)) != vocab_size
        ):
            raise ValueError("Invalid model vocabulary.")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocabulary = list(vocabulary) if vocabulary is not None else None
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm_layers = [
            layers.LSTM(hidden_size, return_sequences=index < num_layers - 1)
            for index in range(num_layers)
        ]
        self.dense = layers.Dense(vocab_size, activation="softmax")

    def build(self, input_shape):
        self.embedding.build(input_shape)
        shape = (*input_shape, self.embedding_dim)
        for layer in self.lstm_layers:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)
        self.dense.build(shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        values = self.embedding(inputs)
        for layer in self.lstm_layers:
            values = layer(values, training=training)
        return self.dense(values)

    def get_config(self):
        return {
            **super().get_config(),
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "vocabulary": self.vocabulary,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NextWordPredictor:
    """Inference independent of Tkinter, also usable by scripts and tests."""

    def __init__(self, model):
        if not getattr(model, "vocabulary", None):
            raise ValueError(
                "This model has no saved vocabulary. Retrain once with --retrain "
                "and the original --text-file."
            )
        self.model = model
        self.vocabulary = model.vocabulary
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}

    def predict_next_word(self, text, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        words = tokenize(text)[-self.model.sequence_length:]
        sequence = [self.word_to_index.get(word, 0) for word in words]
        sequence = [index for index in sequence if index != 0]
        if not sequence:
            return []
        # Right padding works with masked LSTMs on both CPU and GPU.
        sequence.extend([0] * (self.model.sequence_length - len(sequence)))
        scores = np.asarray(self.model(np.array([sequence], dtype=np.int32), training=False))
        if scores.shape != (1, len(self.vocabulary)) or not np.isfinite(scores).all():
            raise ValueError("Model returned invalid prediction scores.")
        probabilities = scores[0]
        # Index 0 is padding and must never be offered as a word.
        indices = np.argsort(-probabilities[1:], kind="stable")[:top_k] + 1
        return [(self.vocabulary[index], float(probabilities[index])) for index in indices]


class NextWordPredictorGUI:
    """Debounce edits and run inference in one worker without blocking Tk."""

    def __init__(self, root, predictor):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.predictor = predictor
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.debounce_id = None
        self.poll_id = None
        self.closed = False
        self.prediction_buttons = []
        root.title("Next Word Predictor")
        root.minsize(560, 220)
        frame = ttk.Frame(root, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Next Word Predictor", font=("Arial", 16, "bold")).pack(pady=8)
        ttk.Label(frame, text="Type a few words, then click a suggestion to continue.").pack()
        self.input_text = tk.StringVar(master=root)
        self.text_entry = ttk.Entry(frame, textvariable=self.input_text, width=60, font=("Arial", 12))
        self.text_entry.pack(fill="x", pady=12)
        self.predictions_frame = ttk.Frame(frame)
        self.predictions_frame.pack(fill="x", pady=6)
        self.status = tk.StringVar(master=root, value="Waiting for input...")
        ttk.Label(frame, textvariable=self.status, wraplength=520).pack(anchor="w", pady=8)
        self.input_text.trace_add("write", self.on_text_changed)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.text_entry.focus_set()

    def clear_predictions(self):
        for button in self.prediction_buttons:
            button.destroy()
        self.prediction_buttons.clear()

    def on_text_changed(self, *_):
        if self.closed:
            return
        if self.debounce_id is not None:
            self.root.after_cancel(self.debounce_id)
            self.debounce_id = None
        self.clear_predictions()
        if not self.input_text.get().strip():
            self.status.set("Waiting for input...")
            return
        self.status.set("Predicting...")
        self.debounce_id = self.root.after(200, self.request_prediction)

    def request_prediction(self):
        self.debounce_id = None
        if self.closed:
            return
        if self.future is not None:
            self.debounce_id = self.root.after(50, self.request_prediction)
            return
        text = self.input_text.get()
        if not text.strip():
            return
        self.future = self.executor.submit(self.predictor.predict_next_word, text)
        self.poll_id = self.root.after(50, self.poll_prediction, text)

    def poll_prediction(self, text):
        from tkinter import ttk

        self.poll_id = None
        if self.closed:
            return
        if not self.future.done():
            self.poll_id = self.root.after(50, self.poll_prediction, text)
            return
        future, self.future = self.future, None
        # A result for a previous input must never replace current suggestions.
        if text != self.input_text.get():
            return
        try:
            predictions = future.result()
        except Exception as error:
            self.status.set(f"Prediction failed: {error}")
            return
        self.clear_predictions()
        for word, probability in predictions:
            button = ttk.Button(
                self.predictions_frame,
                text=f"{word} ({probability:.1%})",
                command=lambda selected=word: self.add_word(selected),
            )
            button.pack(side="left", padx=4)
            self.prediction_buttons.append(button)
        self.status.set("Choose a suggestion or keep typing." if predictions
                        else "Try a word from the training corpus.")

    def add_word(self, word):
        text = self.input_text.get().rstrip()
        self.input_text.set(f"{text} {word}" if text else word)
        self.text_entry.icursor("end")
        self.text_entry.focus_set()

    def close(self):
        self.closed = True
        for callback_id in (self.debounce_id, self.poll_id):
            if callback_id is not None:
                self.root.after_cancel(callback_id)
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def train_model(model, dataset, epochs=10, batch_size=64):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    inputs, targets = dataset.prepare_training_data()
    print(f"Training on {len(inputs)} examples with {len(dataset.vocabulary)} tokens.")
    model.fit(inputs, targets, epochs=epochs, batch_size=batch_size,
              validation_split=0.1 if len(inputs) >= 10 else 0.0)
    return model


def create_and_train_model(dataset, epochs=10, batch_size=64,
                           embedding_dim=100, hidden_size=128, num_layers=3):
    model = LSTMModel(
        vocab_size=len(dataset.vocabulary),
        vocabulary=dataset.vocabulary,
        sequence_length=dataset.sequence_length,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return train_model(model, dataset, epochs=epochs, batch_size=batch_size)


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, default=DEFAULT_CORPUS,
                        help="TXT/DOCX corpus used only when training")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL,
                        help="saved .keras model (includes its vocabulary)")
    parser.add_argument("--sequence-length", type=positive_int, default=10,
                        help="number of context words when training (default: 10)")
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--embedding-dim", type=positive_int, default=100)
    parser.add_argument("--hidden-size", type=positive_int, default=128)
    parser.add_argument("--num-layers", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain", action="store_true", help="replace the saved model by training again")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true", help="prepare the saved model without opening the GUI")
    mode.add_argument("--predict", metavar="TEXT", help="print suggestions without opening the GUI")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model_path.suffix != ".keras":
        parser.error("--model-path must end in .keras")
    try:
        if args.model_path.exists() and not args.retrain:
            print(f"Loading {args.model_path}")
            model = tf.keras.models.load_model(args.model_path, compile=False)
        else:
            tf.keras.utils.set_random_seed(args.seed)
            dataset = TextDataset(args.text_file, args.sequence_length)
            model = create_and_train_model(
                dataset, epochs=args.epochs, batch_size=args.batch_size,
                embedding_dim=args.embedding_dim, hidden_size=args.hidden_size,
                num_layers=args.num_layers,
            )
            args.model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.model_path)
            print(f"Saved model and vocabulary to {args.model_path}")
        predictor = NextWordPredictor(model)
        if args.train_only:
            return 0
        if args.predict is not None:
            predictions = predictor.predict_next_word(args.predict)
            for word, probability in predictions:
                print(f"{word}\t{probability:.1%}")
            if not predictions:
                print("No suggestions: enter at least one word from the training corpus.")
            return 0

        import tkinter as tk

        try:
            root = tk.Tk()
        except tk.TclError as error:
            raise ValueError("Cannot open a Tk window. Use --predict or --train-only without a display.") from error
        NextWordPredictorGUI(root, predictor)
        root.mainloop()
        return 0
    except (OSError, ValueError, tf.errors.OpError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
