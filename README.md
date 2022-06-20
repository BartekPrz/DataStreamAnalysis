# Improving classification of imbalanced and concept drifting data streams

This repository contains all necessary files which were used in my master thesis project

## Contents

The repository is divided into the following directories:

- `Algorithms` - this folder contains source code of all algorithms which are presented in the thesis, source code (written in Java language) is compatible with the MOA (Massive Online Analysis) library
- `OtherAlgorithms` - this folder contains source code of all algorithms which were tested during thesis preparation, ultimately, these approaches were not described in the paper
- `Results` - in this directory you can find all algorithms results which are presented in the thesis - mainly csv files containing classifier performance values over time and classifier performance values averaged over entire streams
- `OtherResults` - this folder containts results from the approaches (`OtherAlgorithms`) that were not described in the paper - content is similar as in the `Results` directory
- `Plots` - in this directory you can find all classifier performance plots - some of them are used in the thesis
- `Scripts` - contains source code use to process results from the `Results` directory, in this directory you can also find batch file which were used to generate experimental results
- `Thesis` - this folder contains source code of the master thesis