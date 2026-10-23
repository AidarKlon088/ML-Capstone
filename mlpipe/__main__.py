
import argparse

from mlpipe.data import run_eda
from mlpipe.train import train
from mlpipe.predict import predict


def main():
    parser = argparse.ArgumentParser(
        description="ML Capstone Pipeline"
    )

    parser.add_argument(
        "command",
        choices=["eda", "train", "predict"]
    )

    parser.add_argument(
        "--input",
        default="data/new_patients.csv"
    )

    parser.add_argument(
        "--output",
        default="reports/predictions.csv"
    )

    args = parser.parse_args()

    if args.command == "eda":
        run_eda()

    elif args.command == "train":
        train()

    elif args.command == "predict":
        predict(args.input, args.output)


if __name__ == "__main__":
    main()
