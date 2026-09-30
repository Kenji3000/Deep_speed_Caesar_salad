#!/usr/bin/env python3
"""Automatic tuning: races the car over and over with different settings and
keeps whatever gives the lowest time with no collisions.

Run INSIDE the container, with nothing else running (it starts its own
simulator, headless, through ./scripts/evaluate.sh):

    python3 /hackathon/race_ws/src/team_driver/tune.py              # default: 40 runs
    python3 /hackathon/race_ws/src/team_driver/tune.py --runs 100 --laps 3

Stop any time with Ctrl+C and run it again later: every result is saved in
results/tuning/log.csv and is reused, so nothing is raced twice.

Method: coordinate search. Starting from the current best settings, try each
parameter one step up and one step down. Keep any change that lowers the
score. When a whole round finds nothing better, halve the steps and go again.

Score = total time (the referee already adds 10 s per collision)
        + 20 s extra per collision, so a clean run always wins,
        or infinity if the run did not complete.
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

REPO = '/hackathon'
OUT = os.path.join(REPO, 'results', 'tuning')
LOG = os.path.join(OUT, 'log.csv')

# name: (start value, step, lowest allowed, highest allowed)
SPACE = {
    'a_lat':   (4.5, 0.5, 2.0, 10.0),
    'a_brake': (4.5, 0.5, 2.0, 10.0),
    'a_accel': (4.0, 0.5, 2.0, 9.5),
    'v_max':   (7.0, 1.0, 3.0, 12.0),
    'margin':  (0.40, 0.05, 0.28, 0.70),
    'ld_base': (0.40, 0.05, 0.20, 1.00),
    'ld_gain': (0.15, 0.03, 0.00, 0.40),
}


def key_of(params):
    return tuple(round(params[k], 4) for k in sorted(SPACE))


def load_log():
    seen = {}
    if os.path.isfile(LOG):
        with open(LOG) as f:
            for row in csv.DictReader(f):
                p = {k: float(row[k]) for k in SPACE}
                seen[key_of(p)] = float(row['score'])
    return seen


def append_log(params, result, score):
    new = not os.path.isfile(LOG)
    with open(LOG, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(sorted(SPACE) + ['score', 'status', 'collisions',
                                        'best_lap', 'total_time', 'result_file'])
        w.writerow([params[k] for k in sorted(SPACE)] + [
            f'{score:.3f}', result.get('status'), result.get('collisions'),
            result.get('best_lap_time'), result.get('total_time'), result.get('_file', '')])


def race(params, laps, evaluate):
    """One headless evaluation with these parameters. Returns (score, result)."""
    tag = time.strftime('%Y%m%dT%H%M%S') + f'_{os.getpid()}_{len(os.listdir(OUT))}'
    yaml_path = os.path.join(OUT, f'params_{tag}.yaml')
    with open(yaml_path, 'w') as f:
        f.write('driver:\n  ros__parameters:\n')
        for k in sorted(params):
            f.write(f'    {k}: {float(params[k])}\n')
    before = set(glob.glob(os.path.join(OUT, '*.json')))
    cmd = [evaluate, '--team', 'tune', '--laps', str(laps), '--warmup', '0',
           '--headless', '--no-build', '--timeout', '900',
           '--driver-params', yaml_path, '--output', OUT]
    with open(os.path.join(OUT, f'run_{tag}.log'), 'w') as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    new = sorted(set(glob.glob(os.path.join(OUT, '*.json'))) - before, key=os.path.getmtime)
    if not new:
        return float('inf'), {'status': 'NO_RESULT'}
    with open(new[-1]) as f:
        result = json.load(f)
    result['_file'] = os.path.basename(new[-1])
    if result.get('status') != 'COMPLETE' or not result.get('scored'):
        return float('inf'), result
    score = float(result['total_time']) + 20.0 * int(result.get('collisions', 0))
    return score, result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=int, default=40, help='new races to run this session')
    ap.add_argument('--laps', type=int, default=3, help='timed laps per race')
    ap.add_argument('--evaluate', default=os.path.join(REPO, 'scripts', 'evaluate.sh'))
    ap.add_argument('--only', default='', help='comma list of parameters to tune, e.g. a_lat,a_brake')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    names = [n for n in SPACE if not args.only or n in args.only.split(',')]
    seen = load_log()
    runs = 0

    def evaluate(p):
        nonlocal runs
        k = key_of(p)
        if k in seen:
            return seen[k]
        if runs >= args.runs:
            raise StopIteration
        runs += 1
        print(f'[{runs}/{args.runs}] racing ' +
              ' '.join(f'{n}={p[n]:g}' for n in sorted(p)), flush=True)
        score, result = race(p, args.laps, args.evaluate)
        append_log(p, result, score)
        seen[k] = score
        print(f'        -> {result.get("status")}, collisions {result.get("collisions")}, '
              f'total {result.get("total_time")}, score {score:.2f}', flush=True)
        return score

    # Start from the best logged settings, or the defaults.
    best = {n: SPACE[n][0] for n in SPACE}
    if os.path.isfile(LOG):
        with open(LOG) as f:
            rows = [r for r in csv.DictReader(f) if float(r['score']) < float('inf')]
        if rows:
            top = min(rows, key=lambda r: float(r['score']))
            best = {n: float(top[n]) for n in SPACE}
    step = {n: SPACE[n][1] for n in SPACE}

    try:
        best_score = evaluate(best)
        while True:
            improved = False
            for n in names:
                for direction in (+1, -1):
                    trial = dict(best)
                    trial[n] = round(min(SPACE[n][3], max(SPACE[n][2], best[n] + direction * step[n])), 4)
                    if trial[n] == best[n]:
                        continue
                    s = evaluate(trial)
                    if s < best_score:
                        best, best_score, improved = trial, s, True
                        print(f'  NEW BEST {best_score:.2f}: {n}={trial[n]:g}', flush=True)
                        break
            if not improved:
                step = {n: step[n] / 2 for n in step}
                if all(step[n] < SPACE[n][1] / 8 for n in names):
                    print('Converged.')
                    break
    except StopIteration:
        print(f'Reached {args.runs} runs.')
    except KeyboardInterrupt:
        print('\nStopped. Progress is saved; run again to continue.')

    best_yaml = os.path.join(OUT, 'best_params.yaml')
    with open(best_yaml, 'w') as f:
        f.write(f'# best score {best_score:.2f} from tune.py\ndriver:\n  ros__parameters:\n')
        for k in sorted(best):
            f.write(f'    {k}: {best[k]}\n')
    print(f'\nBest score {best_score:.2f} with:')
    for k in sorted(best):
        print(f'  {k}: {best[k]}')
    print(f'Saved to {best_yaml}')


if __name__ == '__main__':
    sys.exit(main())
