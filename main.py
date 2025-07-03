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

MODEL_CLASSES = {
    "lstm": JointPunctCapitalModel,
    "rnn": JointPunctCapitalRNN,
}

def main(mode: str, model_type: str, bidirectional: bool):
    train_sentences, val_sentences, test_sentences = split_data_from_file(
        "es_419_validas.txt",
        train_ratio=0.8,
        val_ratio=0.1,
        max_lines=1000
    )

    if mode == "train":
        print(f"Training on {len(train_sentences)} instances, validating on {len(val_sentences)} instances")

        if model_type == "classic":
            classic_model = ClassicPunctuationCapitalizationModel()
            classic_model.train(train_sentences)
            classic_model.save_model("classic_punct_model.pkl")

        else:
            model_cls = MODEL_CLASSES.get(model_type)
            if model_cls is None:
                raise ValueError(f"Unknown model type: {model_type}")
            
            rnn_model = RNNPunctuationCapitalizationModel(
                model_cls=model_cls,
                bidirectional=bidirectional,
            )
            rnn_model.train(train_sentences, val_sentences, epochs=10)
            filename = f"trained_{'bi' if bidirectional else ''}{model_type}_model.pt"
            rnn_model.save_model(filename)

    elif mode == "test":
        print(f"Testing on {len(test_sentences)} instances")

        if model_type == "classic":
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

        else:
            model_cls = MODEL_CLASSES.get(model_type)
            if model_cls is None:
                raise ValueError(f"Unknown model type: {model_type}")

            rnn_model = RNNPunctuationCapitalizationModel(
                model_cls=model_cls,
                bidirectional=bidirectional,
            )
            filename = f"trained_{'bi' if bidirectional else ''}{model_type}_model.pt"
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
        "--model_type",
        choices=["classic", "rnn", "lstm"],
        default="lstm",
        help="Type of model to use"
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="Use bidirectional RNN/LSTM (ignored for classic model)"
    )
    args = parser.parse_args()
    main(args.mode, args.model_type, args.bidirectional)
