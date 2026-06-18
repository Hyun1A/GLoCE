"""Compute Acc_t / Acc_r / H_cc from the success_prompts.csv produced by
evaluate_giphy_fidelity.py.

Acc_t = fraction of images where the *erased target* celeb is still GCD-top1 (lower=better).
Acc_r = fraction of images where the *remaining* celeb is detected (>0.9) (higher=better).
H_cc  = harmonic mean of efficacy (1-Acc_t) and specificity (Acc_r) [paper Eq.10].
"""
import sys, pandas as pd

def main(csv_path):
    df = pd.read_csv(csv_path)
    n = len(df)
    ft = df['find_t'].astype(str).str.lower().isin(['true', '1', '1.0'])
    fp = df['find_p'].astype(str).str.lower().isin(['true', '1', '1.0'])
    acc_t = 100.0 * ft.sum() / n
    acc_r = 100.0 * fp.sum() / n
    eff = 1.0 - acc_t / 100.0
    spec = acc_r / 100.0
    h = 0.0 if (eff == 0 or spec == 0) else 2.0 / (1.0/eff + 1.0/spec) * 100.0
    print(f"Total images: {n}")
    print(f"Acc_t: {acc_t:.2f}")
    print(f"Acc_r: {acc_r:.2f}")
    print(f"H_cc: {h:.2f}")

if __name__ == '__main__':
    main(sys.argv[1])
