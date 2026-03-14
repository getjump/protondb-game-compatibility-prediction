"""Phase 19: Test LLM-only verdicts (no rule-based) effect on training."""
from __future__ import annotations
import argparse, logging, time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/protondb.db")
    args = parser.parse_args()

    from protondb_settings.db.connection import get_connection
    from protondb_settings.ml.features.embeddings import load_embeddings
    from protondb_settings.ml.train import _build_feature_matrix, _time_based_split
    from protondb_settings.ml.relabeling import get_relabel_ids
    from protondb_settings.ml.irt import fit_irt, add_irt_features, contributor_aware_relabel, add_error_targeted_features
    from protondb_settings.ml.models.cascade import train_stage1, train_stage2, CascadeClassifier
    from protondb_settings.ml.models.classifier import CATEGORICAL_FEATURES

    conn = get_connection(args.db)
    emb_data = load_embeddings(Path(args.db).parent / "embeddings.npz")

    # Load LLM-only verdicts (exclude rule-based)
    llm_verdicts = {}
    for r in conn.execute("SELECT report_id, verdict FROM inferred_verdicts WHERE reason NOT LIKE 'rule:%'").fetchall():
        llm_verdicts[r["report_id"]] = r["verdict"]
    logger.info("LLM-only verdicts: %d", len(llm_verdicts))

    X, y_raw, ts, rids, lm = _build_feature_matrix(conn, emb_data)
    X_train, X_test, y_train_raw, y_test, train_rids, test_rids = _time_based_split(
        X, y_raw, ts, 0.2, report_ids=rids)
    relabel_ids = get_relabel_ids(conn)
    theta, difficulty = fit_irt(conn)
    X_train = add_irt_features(X_train, train_rids, conn, theta, difficulty)
    X_test = add_irt_features(X_test, test_rids, conn, theta, difficulty)
    X_train = add_error_targeted_features(X_train, train_rids, conn)
    X_test = add_error_targeted_features(X_test, test_rids, conn)
    conn.close()

    for col in CATEGORICAL_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    conn2 = get_connection(args.db)
    y_baseline, _ = contributor_aware_relabel(y_train_raw, train_rids, relabel_ids, conn2, theta)
    conn2.close()

    def eval_model(y_tr, label):
        s1 = train_stage1(X_train, y_tr, X_test, y_test)
        s2, drops = train_stage2(X_train, y_tr, X_test, y_test)
        cas = CascadeClassifier(s1, s2, drops)
        y_pred = cas.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        per = f1_score(y_test, y_pred, average=None)
        oob_r = (y_pred[y_test == 2] == 2).mean()
        print(f"  {label:40s} F1={f1:.4f} b={per[0]:.3f} t={per[1]:.3f} o={per[2]:.3f} oob_r={oob_r:.3f}")
        return f1

    # A. Baseline
    print("\nA. BASELINE")
    f1_bl = eval_model(y_baseline, "baseline")

    # B. LLM-only fix (no rule-based)
    print("\nB. LLM-ONLY FIX")
    y_llm = y_baseline.copy()
    n_fix = 0
    for i, rid in enumerate(train_rids):
        if rid in llm_verdicts and y_llm[i] == 1:
            if llm_verdicts[rid] == "works_oob":
                y_llm[i] = 2
                n_fix += 1
    logger.info("LLM fix: %d tinkering → works_oob in train", n_fix)

    for cls, name in [(0, "borked"), (1, "tinkering"), (2, "works_oob")]:
        n_bl = (y_baseline == cls).sum()
        n_new = (y_llm == cls).sum()
        print(f"  {name:12s}: {n_bl:6d} → {n_new:6d} ({n_new-n_bl:+d})")

    f1_llm = eval_model(y_llm, "llm_only_fix")
    print(f"\n  Delta: {f1_llm - f1_bl:+.4f} F1")

if __name__ == "__main__":
    main()
