# Statistics Cookbook

Barebones introduction to statistical data analysis techniques, aimed at undergraduates doing research for the first time. This material in no way substitutes for a proper course, but hopefully provides enough to get started.

## Contents

* [Basic model fitting](basic_model_fitting/): just enough to get started with **maximum likelihood methods** for fitting model parameters, finding confidence intervals, evaluating goodness of fit, and comparing competing models.

## Avoiding `git` conflicts

Before running `git pull` to update anything:
* Run `git status` to list any locally modified files.
* For each modified file,
    * if you want to keep your local changes,
        * make a copy, e.g. `cp thisFile.ipynb thisFile_mine.ipynb`;
    * run `git checkout -- thisFile.ipynb`.

Now it should be safe to `git pull`.

## (Some) other resources

* [Physics 366](https://github.com/KIPAC/StatisticalMethods): graduate/advanced undergraduate course in statistical methods in astrophysics, mostly Bayesian analysis
* [LSSTC Data Science Fellowship Program](https://github.com/LSSTC-DSFP/LSSTC-DSFP-Sessions), especially sessions 4 and 10

## Author

[Adam Mantz](https://github.com/abmantz). Any errors or unnecessary snarkiness is his fault.

## License

All materials Copyright 2020 the authors.

Unless otherwise noted, all content is licensed under Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).
