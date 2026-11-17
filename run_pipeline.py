"""Run the AdaptiveAML pipeline end to end.

    python run_pipeline.py                 # all stages on data/raw (the real SynthAML files)
    python run_pipeline.py --sample        # all stages on the small test fixture (NOT real results)
    python run_pipeline.py --stage 4 5 6   # only some stages
    python run_pipeline.py --stage 8 9 10  # behavioural profiles, sequence models, feature lab
    python run_pipeline.py --stage 11 12 13  # ensembles, final comparison, export for the web dashboard
"""
import argparse
import importlib
import time

from adaptiveaml.common import load_config, seed_everything

STAGES = {1: "s01_ingest", 2: "s02_features", 3: "s03_eda", 4: "s04_regression",
          5: "s05_classification", 6: "s06_evaluate", 7: "s07_explain",
          8: "s08_clustering", 9: "s09_sequences", 10: "s10_compare",
          11: "s11_ensembles", 12: "s12_final", 13: "s13_export"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, nargs="*", default=list(STAGES))
    ap.add_argument("--sample", action="store_true", help="use the synthetic test fixture in data/sample")
    args = ap.parse_args()

    cfg = load_config()
    if args.sample:
        cfg["data"]["alerts_csv"] = "data/sample/synthetic_alerts.csv"
        cfg["data"]["transactions_csv"] = "data/sample/synthetic_transactions.csv"
        cfg["data_source"] = "fixture"
    for k in args.stage:
        seed_everything(cfg["seed"])
        t0 = time.time()
        importlib.import_module(f"adaptiveaml.{STAGES[k]}").run(cfg)
        print(f"    stage {k} done in {time.time() - t0:.1f}s")
