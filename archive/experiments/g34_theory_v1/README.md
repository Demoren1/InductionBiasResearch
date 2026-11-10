# G34: theory-derived proposals

> Архивировано 8 ноября 2026. Сырые results/ удалены. [Сохранённый аудит и восстановление](../../2026-11-08/README.md). Ниже — исторические команды: перед новым обучением нужно вернуть пакеты в experiments/ и заново подготовить данные.

This isolated experiment compares the immutable G34 with two alternatives:
diversity within its existing pool, and additional repeated-pair log-ratio,
positive-log calibration and sign-profile proposals. The source refit budget
is 32 candidates plus the same anchors for every method. Proposal compute
cost is measured separately.

The initial [protocol](PROTOCOL.md) defines source-only development and fresh
confirmation on Pattern and synthetic DeepSets. The general synthetic
extension is described below. These experiments do not cover real data.

## Execution

Use a new output directory. The setup action rejects an existing one.
It exports only old source observations for development and generates new
confirmation families with a separately recorded seed.

```bash
python -m experiments.g34_theory_v1.run --base NEW_ROOT --method setup
python -m experiments.paper_structure_benchmark_v1.execution.fleet \
  --plan NEW_ROOT/baseline_plan.json --per-gpu 1
```

Construct development proposals for case_00 and case_03, condition1:

```bash
python -m experiments.g34_theory_v1.variant \
  --root NEW_ROOT/development --case case_00 --condition 1 --phase proposals
python -m experiments.g34_theory_v1.variant \
  --root NEW_ROOT/development --case case_03 --condition 1 --phase proposals
python -m experiments.g34_theory_v1.run \
  --base NEW_ROOT --method plan-development
python -m experiments.paper_structure_benchmark_v1.execution.fleet \
  --plan NEW_ROOT/development_plan.json --per-gpu 1
```

After inspecting development source selection, freeze the method and
evaluate the fixed alternatives on confirmation sources:

```bash
python -m experiments.g34_theory_v1.run --base NEW_ROOT --method register
python -m experiments.g34_theory_v1.run \
  --base NEW_ROOT --method plan-confirmation
python -m experiments.paper_structure_benchmark_v1.execution.fleet \
  --plan NEW_ROOT/variants_plan.json --per-gpu 1
python -m experiments.g34_theory_v1.run --base NEW_ROOT --method export
python -m experiments.paper_structure_benchmark_v1.execution.fleet \
  --plan NEW_ROOT/target_plan.json --per-gpu 1
python -m experiments.g34_theory_v1.report --base NEW_ROOT --phase freeze
python -m experiments.g34_theory_v1.report --base NEW_ROOT --phase score
```

Run one fleet at a time. CPU proposal intervals release a subprocess's CUDA
context; another scheduler must not treat that reserved GPU as available.
Only GPUs 0–5 are used. On the original host CUDA requires execution outside
the filesystem sandbox.

The one-time method registration checks bank.py and variant.py before each
confirmation learner process. Query answers and structural truth are opened
only by the scorer after all 42 target prediction files are frozen.
The report retains per-task metrics, independent-family comparisons,
train/selection curves, recovery/recall and costs.

## General synthetic extension

The [general protocol](GENERAL_PROTOCOL.md) adds local/quartet_mix/random with
free neuron heads, common/independent bias strata and source noise0/.2.
The method must match the existing Pattern/DeepSets registration; no new
development or tuning is performed. The original report and artifacts remain
separate.

Completed results: [general synthetic report](/home/udeneev-av/InductionBiasResearch/G34_GENERAL_SYNTHETIC_REPORT_2026-11-08.md).

```bash
python -m experiments.g34_theory_v1.general \
  --base NEW_GENERAL_ROOT --previous FINISHED_PATTERN_DEEPSETS_ROOT
python -m experiments.paper_structure_benchmark_v1.execution.fleet \
  --plan NEW_GENERAL_ROOT/baseline_plan.json --per-gpu 1
python -m experiments.paper_structure_benchmark_v1.execution.fleet \
  --plan NEW_GENERAL_ROOT/variants_plan.json --per-gpu 1
python -m experiments.g34_theory_v1.run --base NEW_GENERAL_ROOT --method export
python -m experiments.paper_structure_benchmark_v1.execution.fleet \
  --plan NEW_GENERAL_ROOT/target_plan.json --per-gpu 1
python -m experiments.g34_theory_v1.general_report \
  --base NEW_GENERAL_ROOT --phase freeze
python -m experiments.g34_theory_v1.general_report \
  --base NEW_GENERAL_ROOT --phase score
```

Use `general_report --phase render` to regenerate figures/text from saved
scoring, without reading query answers again or repeating training.
