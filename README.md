# jax_parametric_models

## Setup

```
cmsrel CMSSW_14_1_0_pre4
cd CMSSW_14_1_0_pre4/src
cmsenv
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v10.1.0
scramv1 b clean; scramv1 b

cd data/tutorials/parametric_exercise
git clone git@github.com:maxgalli/jax_parametric_models.git
git clone git@github.com:maxgalli/StatsStudies.git
```

### Setup with `uv`

```shell
uv sync
uv run part1_2.py
```

## Links

- [combine parametric fit tutorial](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/tutorial2023/parametric_exercise/#session-structure)
- [zfit version](https://github.com/maxgalli/StatsStudies/tree/master/ExercisesForCourse/Hgg_zfit) to get stuff from
