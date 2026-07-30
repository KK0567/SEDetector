# -*- coding: utf-8 -*-
"""
DAPT2020 Privacy Attack - One-click Runner
Usage: python run_all_new.py
"""
import sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from privacy_attack_framework_new import load_dapt, run_attack, print_summary, save_csv

if __name__ == "__main__":
    print("=" * 60)
    print("  DAPT2020 Privacy Attack Experiment")
    print("  reconstruction / semantic_inference / composition")
    print("  Raw / BSS / SEU  x  LR / RF / MLP")
    print("  5-fold stratified CV")
    print("=" * 60)

    t0 = time.time()
    records = load_dapt()

    all_results = []
    for attack in ["reconstruction", "semantic_inference", "composition"]:
        print(f"\n  >>> {attack}")
        res = run_attack(attack, records, n_folds=5, max_samples=30000)
        all_results.extend(res)

    print_summary(all_results)

    csv_path = os.path.join(HERE, "privacy_attack_results_new.csv")
    save_csv(all_results, csv_path)

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed/60:.1f} min")
