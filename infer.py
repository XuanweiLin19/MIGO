import argparse
from migo.registry import run_task


def parse_args():
    parser = argparse.ArgumentParser(description="Unified inference entrypoint for MIGO.")
    parser.add_argument("--task", required=True, choices=["scRNA_scHiC", "RNA_Ribo_seq", "scRNA_ADT", "scRNA_scATAC"])
    parser.add_argument("--config", default=None, help="Path to JSON config")
    parser.add_argument("--resolution", choices=["1M", "50K"], help="scRNA_scHiC resolution")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0 or cpu")
    parser.add_argument("--rna-h5ad", dest="rna_h5ad", default=None, help="RNA h5ad path override")
    parser.add_argument("--atac-h5ad", dest="atac_h5ad", default=None, help="ATAC h5ad path override")
    parser.add_argument("--adt-h5ad", dest="adt_h5ad", default=None, help="ADT h5ad path override")
    parser.add_argument("--ribo-h5ad", dest="ribo_h5ad", default=None, help="Ribo h5ad path override")
    parser.add_argument("--hic-h5ad", dest="hic_h5ad", default=None, help="HiC h5ad path override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--num-workers", type=int, default=None, help="Dataloader worker count override")
    parser.add_argument("--train-batch-size", type=int, default=None, help="scRNA_scHiC train batch size override")
    parser.add_argument("--val-batch-size", type=int, default=None, help="scRNA_scHiC val batch size override")
    parser.add_argument("--model-save-path", dest="model_save_path", default=None, help="Model save directory override")
    parser.add_argument("--log-dir", dest="log_dir", default=None, help="Log directory override")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {
        "seed": args.seed,
        "device": args.device,
        "rna_h5ad": args.rna_h5ad,
        "atac_h5ad": args.atac_h5ad,
        "adt_h5ad": args.adt_h5ad,
        "ribo_h5ad": args.ribo_h5ad,
        "hic_h5ad": args.hic_h5ad,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "model_save_path": args.model_save_path,
        "log_dir": args.log_dir,
    }
    run_task(args.task, "infer", config_path=args.config, resolution=args.resolution, overrides=overrides)


if __name__ == "__main__":
    main()

