"""Create synthetic inputs only; no real entities or biological associations."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ncsp_ip.features import vectorize, annotation


def main():
    output = Path('data/synthetic')
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Synthetic output directory already contains files.')
    output.mkdir(parents=True, exist_ok=True)
    names = [f'synthetic_RNA_{i:02d}' for i in range(12)]
    A = np.zeros((12, 6), dtype=np.float32)
    for i in range(10):
        A[i, i % 6] = 1
    x, _, _ = vectorize(names, A)
    ann = np.array([annotation(name) for name in names], dtype=np.float32)
    negative = np.array([(i, (i+2) % 6) for i in range(10)], dtype=np.int64)
    np.savez_compressed(output / 'inputs.npz', A=A, x=x, ann=ann)
    np.savez_compressed(output / 'negative_pairs.npz', pairs=negative)
    print('Created synthetic inputs in data/synthetic.')


if __name__ == '__main__':
    main()
