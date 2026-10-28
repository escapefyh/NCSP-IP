"""Reconstruct scores from a matching decoder checkpoint and context."""
import argparse
from pathlib import Path
import numpy as np
import torch
from ncsp_ip import decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--rna-id', type=int)
    parser.add_argument('--disease-id', type=int)
    args = parser.parse_args()
    decoder.DEV = 'cpu'
    torch.set_num_threads(2)
    with np.load(args.model_dir / 'context.npz', allow_pickle=False) as z:
        context = {k:z[k] for k in ['A','x','ann','G','prior']}
    states = torch.load(args.model_dir / 'models.pt', map_location='cpu', weights_only=True)
    scores = decoder.predict(context, states)
    if args.rna_id is None or args.disease_id is None:
        parser.error('Supply both --rna-id and --disease-id (zero-based catalogue indices).')
    if not (0 <= args.rna_id < scores.shape[0] and 0 <= args.disease_id < scores.shape[1]):
        parser.error('Catalogue index out of range.')
    print(float(scores[args.rna_id, args.disease_id]))


if __name__ == '__main__':
    main()
