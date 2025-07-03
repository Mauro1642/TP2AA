import pandas as pd
import torch
import random
import argparse

from classic import ClassicPunctuationCapitalizationModel, evaluate_model
from rnn import (
    RNNPunctuationCapitalizationModel,
    JointPunctCapitalModel,
    JointPunctCapitalRNN,
    evaluate_model_rnn,
)
from utils import split_data_from_file

RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def main(mode: str, which: str):
    train_sentences, val_sentences, test_sentences = split_data_from_file(
        "es_419_validas.txt",
        train_ratio=0.8,
        val_ratio=0.1,
        max_lines=1000
    )

    if mode == "train":
        print(f"Training on {len(train_sentences)} instances, validating on {len(val_sentences)} instances")

        if which in ("classic", "both"):
            classic_model = ClassicPunctuationCapitalizationModel()
            classic_model.train(train_sentences)
            classic_model.save_model("classic_punct_model.pkl")

        if which in ("rnn", "birnn", "both"):
            bidirectional = (which == "birnn")
            model_class = JointPunctCapitalModel if bidirectional else JointPunctCapitalRNN
            rnn_model = RNNPunctuationCapitalizationModel(
                model_cls=model_class,
                bidirectional=bidirectional,
            )
            rnn_model.train(train_sentences, val_sentences, epochs=10)
            filename = "trained_birnn_model.pt" if bidirectional else "trained_rnn_model.pt"
            rnn_model.save_model(filename)

    elif mode == "test":
        print(f"Testing on {len(test_sentences)} instances")

        if which in ("classic", "both"):
            classic_model = ClassicPunctuationCapitalizationModel()
            classic_model.load_model("classic_punct_model.pkl")
            evaluate_model(classic_model, classic_model._prepare_data(test_sentences))
            for text in [
                "pasado mañana",
                "estás asustado",
                "cindy espero que estes muy orgullosa de lo que haz hecho",
                "cómo estás"
            ]:
                print(classic_model.predict_and_reconstruct(text))

        if which in ("rnn", "birnn", "both"):
            bidirectional = (which == "birnn")
            model_class = JointPunctCapitalModel if bidirectional else JointPunctCapitalRNN
            rnn_model = RNNPunctuationCapitalizationModel(
                model_cls=model_class,
                bidirectional=bidirectional,
            )
            filename = "trained_birnn_model.pt" if bidirectional else "trained_rnn_model.pt"
            rnn_model.load_model(filename)

            # Predict CSV
            input_csv = "datos_test.csv"
            output_csv = "predicted_output.csv"
            input_df = pd.read_csv(input_csv)
            rnn_model.predict_and_fill_csv(input_df, output_file=output_csv)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'test'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test punctuation-capitalization models")
    parser.add_argument(
        "mode",
        choices=["train", "test"],
        help="Mode: train models or test existing models"
    )
    parser.add_argument(
        "--model",
        choices=["classic", "rnn", "birnn", "both"],
        default="both",
        help="Which model(s) to train/test (default: both)"
    )
    args = parser.parse_args()
    main(args.mode, args.model)
