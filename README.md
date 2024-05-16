## Applied modeling with BART: How, When, and Why

All material from the workshop on BART held at the [2024 CMSAC Football Workshop](https://www.stat.cmu.edu/cmsac/footballworkshop/) on 18 May 2024.

The direcotires `bart_presentation` and `catch_probability` contain the code (as RMarkdown documents) for the synthetic data and catch probability examples.

## Installation and setup

In the workshop, we used the package **BART** (available on CRAN) and **flexBART**, which is a re-implementation of BART that handle categorical predictors in a more thoughtful fashion.
You can install **BART** directly from CRAN:
```
install.packages("BART")
```
To install, **flexBART** you will need to use `devtools::install_github`:
```
devtools::install_github(repo = "skdeshpande91/flexBART/", subdir = "flexBART")
```

If you are using macOS, you will need to have previously set up the macOS toolchain for R.
This is typically something you have to do only once (and if you have used packages with **Rcpp** dependence, you probably already have done this).
But in case you haven't and are unable to install **BART** and/or **flexBART**, you might find the following links helpful:

+ [R for macOS](https://cran.r-project.org/bin/macosx/tools/) for more information.
+ [Instructions from The Coatless Professor](https://thecoatlessprofessor.com/programming/cpp/r-compiler-tools-for-rcpp-on-macos/). Note that these instructions may not be relevant for later versions of macOS (e.g. ones shipped with an Apple silicon processor)
+ [This StackOverflow post](https://stackoverflow.com/questions/69639782/installing-gfortran-on-macbook-with-apple-m1-chip-for-use-in-r/72997915#72997915), which outlines how to install the necessary C++ compiler, gfortran, and set the necessary paths.

