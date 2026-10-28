"""Train the adopted decoder from a TRAIN-only context, optionally fitting its prior."""
import argparse
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from ncsp_ip import decoder
from ncsp_ip.prior import normalized, train_prior


def validate(A, x, ann, negative):
    if A.ndim != 2 or not np.isin(A, [0, 1]).all() or not A.any():
        raise ValueError('A must be a nonempty binary TRAIN-positive matrix.')
    if x.ndim != 2 or len(x) != len(A) or ann.shape != (len(A), 92):
        raise ValueError('Expected x: (RNA, features), ann: (RNA, 92).')
    if not all(np.isfinite(v).all() for v in [A, x, ann]):
        raise ValueError('Input arrays must be finite.')
    if negative.ndim != 2 or negative.shape[1] != 2 or len(negative) != int(A.sum()):
        raise ValueError('The adopted model requires one proxy negative per TRAIN positive.')
    if (negative < 0).any() or (negative[:, 0] >= A.shape[0]).any() or (negative[:, 1] >= A.shape[1]).any():
        raise ValueError('Negative pair index is outside A.')
    if len(np.unique(negative, axis=0)) != len(negative) or A[negative[:, 0], negative[:, 1]].any():
        raise ValueError('Proxy negatives must be unique and disjoint from TRAIN positives.')
    if not np.array_equal(np.bincount(negative[:, 0], minlength=len(A)), A.sum(1)):
        raise ValueError('Negatives must match the TRAIN-positive count of each RNA.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--negative', type=Path, required=True)
    parser.add_argument('--teachers', type=Path, help='Optional original frozen teacher checkpoint.')
    parser.add_argument('--output', type=Path, default=Path('outputs/run'))
    parser.add_argument('--seed', type=int, default=2026111930)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose an empty output directory to preserve existing models.')
    from ncsp_ip import prior as prior_module
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable.')
    decoder.DEV = prior_module.DEVICE = args.device
    torch.set_num_threads(2)
    with np.load(args.input, allow_pickle=False) as z:
        A, x, ann = [z[k].astype(np.float32) for k in ['A', 'x', 'ann']]
        cached_prior = z['prior'].copy() if 'prior' in z else None
        P = z['P'].astype(np.float32) if 'P' in z else None
    with np.load(args.negative, allow_pickle=False) as z:
        raw_negative = z['pairs']
        if raw_negative.dtype.kind not in 'iu':
            raise ValueError('Negative indices must be integers.')
        negative = raw_negative.astype(np.int64)
    validate(A, x, ann, negative)
    if args.teachers:
        if cached_prior is None:
            parser.error('--teachers requires a matching prior array in --input.')
        bundle = torch.load(args.teachers, map_location='cpu', weights_only=True)
        prior = cached_prior
    else:
        prior, bundle = train_prior(A, x, ann, args.seed, P)
    if prior.shape != A.shape or not np.isfinite(prior).all() or not ((prior >= 0) & (prior <= 1)).all():
        raise ValueError('Invalid prior probability matrix.')
    context = dict(A=A, x=x, ann=ann, G=normalized(A), prior=prior, bundle=bundle, arms=['off', 'real'])
    states, scores = {}, []
    for offset in [101, 202, 303]:
        state, score = decoder.fit(dict(seed=args.seed), 'full', offset, context, negative)
        states[str(offset)] = state
        scores.append(score)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(states, args.output / 'models.pt')
    np.savez_compressed(args.output / 'context.npz', **{k:context[k] for k in ['A', 'x', 'ann', 'G', 'prior']})
    np.savez_compressed(args.output / 'scores.npz', independent=np.mean(scores, axis=0), seed_scores=scores)
    (args.output / 'run.json').write_text(json.dumps(dict(model='NCSP-IP lambda=0', seed=args.seed, offsets=[101,202,303], device=args.device, epochs=100, residual_cap=2, temperature=5, parameter_regularization=1e-4, synthetic_or_user_supplied_data=True), indent=2), encoding='utf-8')
    print('Saved trained decoder and predictions.')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
