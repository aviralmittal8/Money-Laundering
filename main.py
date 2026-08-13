import argparse
import os

from src.train import run_training
from src.predict import predict_from_file


def main():
    parser = argparse.ArgumentParser(description="Fraud detection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--data", required=True, help="Path to CSV or Excel dataset")
    train_parser.add_argument("--outputs", default="outputs", help="Outputs directory")
    train_parser.add_argument("--models", default="models", help="Models directory")
    train_parser.add_argument("--hash-dim", type=int, default=128, help="Hashing dimension for categorical features")
    train_parser.add_argument("--gnn-edge-sample", type=int, default=200000, help="Edges sampled per GNN epoch (0 = all)")
    train_parser.add_argument("--gnn-infer-batch", type=int, default=8192, help="Batch size for GNN inference")
    train_parser.add_argument("--epochs-ann", type=int, default=12, help="ANN epochs")
    train_parser.add_argument("--epochs-gnn", type=int, default=8, help="GNN epochs")
    train_parser.add_argument("--batch-size", type=int, default=1024, help="ANN batch size")
    train_parser.add_argument("--min-precision", type=float, default=0.15, help="Minimum precision target during ensemble threshold tuning")
    train_parser.add_argument("--eval-data", help="Optional evaluation dataset path. If omitted, uses --data")
    train_parser.add_argument("--target-recall", type=float, help="Optional target recall for ensemble threshold selection")

    predict_parser = subparsers.add_parser("predict", help="Batch predict")
    predict_parser.add_argument("--data", required=True, help="Path to CSV or Excel dataset")
    predict_parser.add_argument("--models", default="models", help="Models directory")

    args = parser.parse_args()

    if args.command == "train":
        metrics = run_training(
            args.data,
            outputs_dir=args.outputs,
            models_dir=args.models,
            epochs_ann=args.epochs_ann,
            epochs_gnn=args.epochs_gnn,
            batch_size=args.batch_size,
            hash_dim=args.hash_dim,
            gnn_edge_sample=args.gnn_edge_sample,
            gnn_infer_batch=args.gnn_infer_batch,
            min_precision=args.min_precision,
            eval_data_path=args.eval_data,
            target_recall=args.target_recall,
        )
        print() # Newline before the "--- Training Complete ---" message
        print('--- Training Complete ---')
        print('Evaluation Metrics (Ensemble):')
        for k, v in metrics["metrics"].items():
            print(f"  {k}: {v:.4f}")
        if "individual_metrics" in metrics:
            print("Individual Model Metrics:")
            for model_name, model_metrics in metrics["individual_metrics"].items():
                print(f"  {model_name.upper()}:")
                print(
                    f"    accuracy={model_metrics['accuracy']:.4f}, "
                    f"precision={model_metrics['precision']:.4f}, "
                    f"recall={model_metrics['recall']:.4f}, "
                    f"f1={model_metrics['f1']:.4f}, "
                    f"auc={model_metrics['auc']:.4f}"
                )
        print(f'Outputs saved to: {metrics["outputs_dir"]}')
        print(f'Models saved to: {metrics["models_dir"]}')
    elif args.command == "predict":
        df, probs = predict_from_file(args.data, models_dir=args.models)
        df = df.copy()
        df["predicted_prob"] = probs
        os.makedirs("outputs", exist_ok=True)
        out_path = "outputs/predictions.csv"
        df.to_csv(out_path, index=False)
        print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
