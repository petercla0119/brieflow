# Config Glossary

The brieflow config holds all of the parameters used for a brieflow run.
Each notebook is used to configure the parameter variables, which are then saved to the `config.yml`.
Each analysis requires a specific `config.yml` and the associated files (pool dataframe, cell classification model, etc).
An example config for the `main` branch is outlined in [tests/small_test_analysis/config/config.yml](https://github.com/cheeseman-lab/brieflow/blob/main/tests/small_test_analysis/config/config.yml).
While all of the parameters are explicity outlined in each notebook, we provide additional comments on some here as well:
- `preprocess:sbs_samples_fp`/`preprocess:phenotype_samples_fp`: Path to dataframes with one entry for an SBS/phenotype file's path and the associated metadata (plate, well, tile, etc).
- `sbs:df_design_path`: Path to dataframe with SBS pool design information regarding gene, sgRNA, oligo, etc.
- `*_combo_fp`: Path to dataframe with wildcards for file processing in a particular module.
Each combination usually corresponds to a one process that needs to be done with Snakemake.
For example, each plate, well, tile combination in `phenotype_combo_fp` corresponds to one raw file conversion during preprocessing.
- `merge:positions_image_qc`: with `merge:approach: positions`, whether each well also writes the seam, phenotype-in-SBS and mosaic image readouts (default `true`). They read a bounded sample of tiles; set `false` when the aligned images and nuclei labels are no longer on disk or to save the extra minute per well.
