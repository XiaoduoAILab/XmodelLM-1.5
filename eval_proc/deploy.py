import argparse

from utils import deploy_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="out/iter-048000")

    args = parser.parse_args()

    deploy_model(args.model)
