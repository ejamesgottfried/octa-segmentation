---
name: oscar-run
description: Write the run card James pastes on Oscar to launch, monitor, debug, or collect results from a SLURM job or Jupyter session. Use when a task needs the cluster, when James pastes a SLURM log or error, asks why a job failed, or asks whether to use sbatch or Jupyter.
---

# Running things on Oscar

Claude Code runs on the laptop and cannot reach Oscar. Everything Oscar-side is a
**run card**: a short, numbered block James copies into an Oscar terminal.

## sbatch or Jupyter?

| Use sbatch (slurm/*.sbatch) | Use Jupyter (Open OnDemand GPU session) |
|---|---|
| Training, any sweep, anything over ~10 min | Looking at predictions, quick plots |
| Test-set evaluation that produces numbers for the paper | Exploring a new dataset's files and shapes |
| Anything you will need to rerun or cite | Debugging one image or one checkpoint |

Rule of thumb: if a number will appear on a poster, it came from a script, not a cell.

## Run card format (always this shape)

````markdown
### Run card: <experiment>
**On the laptop**
```bash
git add -A && git commit -m "<message>" && git push
```
**On Oscar** (terminal, not a notebook cell)
```bash
cd /users/egottfri/code/octa-segmentation
git pull
mkdir -p logs
python scripts/<name>.py --list-tasks          # expect N tasks
sbatch --array=0 slurm/<name>.sbatch           # one test task first
```
**Check the test task** (a few minutes in)
```bash
squeue -u egottfri
tail -n 30 logs/<name>-<JOBID>_0.out
```
Looks healthy (loss falling, no errors)? Submit the rest:
```bash
sbatch --array=1-<N-1>%8 slurm/<name>.sbatch
```
**Expected:** <runtime per task>, outputs in results/<name>/<config_id>/.
**Next step after it finishes:** <evaluate-run command>.
````

## Monitoring commands

```bash
squeue -u egottfri                                   # what is queued / running
sacct -j <JOBID> --format=JobID%20,State,Elapsed,MaxRSS,ExitCode   # finished tasks
tail -f logs/<name>-<JOBID>_<TASK>.out               # live log (Ctrl-C to stop)
grep -l "Error\|Traceback" logs/<name>-<JOBID>_*.err # which tasks crashed
scancel <JOBID>                                      # cancel everything in an array
```
Resubmitting after a crash or timeout is safe: finished configs are skipped.
Resubmit only the failed task ids: `sbatch --array=3,7 slurm/<name>.sbatch`.

## Diagnosing a failure (James pastes the .err tail)

| Symptom in log | Likely cause | Fix |
|---|---|---|
| No log file at all, job vanished | `logs/` did not exist | `mkdir -p logs`, resubmit |
| `conda: command not found` / env not active | missing `source .../conda.sh` | copy header from template |
| `ModuleNotFoundError: train` | script not adding `src/` to path, or run from wrong dir | `cd` repo root; check `sys.path.insert` |
| `CUDA out of memory` | batch too big for that GPU | lower `batch_size` or request `--constraint` for larger GPU |
| `DUE TO TIME LIMIT` | underestimated `-t` | resubmit (resumes), raise `-t`, log runtime in RESULTS.md |
| `FileNotFoundError` on data | path changed or wrong split name | fix in `src/paths.py`, never in the script |
| `not divisible by 16` | image size vs 4 pooling stages | pad/crop in dataset, do not resize labels |
| Many tasks pending forever | GPU partition busy | lower `%8` throttle, shorter `-t` gets scheduled sooner |

Ask James for `sacct` output and the last 30 lines of `.err` before guessing.

## Bringing results back to the laptop

Only small summaries travel through git. Checkpoints stay on Oscar.
```bash
# on Oscar, after evaluation
python scripts/evaluate_experiment.py <experiment>     # writes reports/<experiment>/*.csv
git add reports/<experiment> && git commit -m "results: <experiment>" && git push
# on the laptop
git pull
```
Never commit anything under results/, data/, or anything derived from ARIAS images.

## Jupyter session tips

- First cell: `%load_ext autoreload`, `%autoreload 2`, then `sys.path.insert(0, "../../src")`.
- After `git pull`, autoreload picks up function edits. Restart the kernel after changing
  a class definition or if something behaves strangely.
- Run git from a terminal tab, never from a notebook cell (a token was leaked that way once).
- Record runtimes you observe in the docs/RESULTS.md run log so time limits stay honest.
