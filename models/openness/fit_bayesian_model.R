#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-


# SETTINGS =====================================================================

library(tidyr)
library(tibble)
library(dplyr)
library(stringr)
library(jsonlite)
library(ggplot2)
library(brms)
library(tidybayes)
library(parallel)


# DATA READ-IN AND PREPARATION =================================================

# read data from file
dat <-
  fromJSON(readLines("omega_time_averaged.json")) %>%
  as_tibble() %>%
  rename(
    omega = omega_mean
  ) %>%
  filter(
    is.na(omega) == FALSE
  ) %>%
  group_by(
    system,
    pdbid,
    ff,
    wm,
    efield
  )

dat <- dat %>% ungroup()


# BAYESIAN REGRESSION ==========================================================

# will sample in parallel
options(mc = parallel::detectCores())

# specify priors
prior <-
  prior(
    normal(0, 1e1),    # zero-field energy difference
    nlpar = "omega0"
  ) +
  prior(
    gamma(2, 0.01),    # E-field coupling (second parameter is rate rather than scale, so inversely proportional to mean)
    nlpar = "m",
    lb = 0.00
  ) +
  prior(
    gamma(2, 100.0),   # intrinsic E-field
    nlpar = "intEfield",
    lb = 0.00
  )

# fit model
fit <-
  brm(
    bf(
      omega ~ 1/(1 + exp(-omega0 -m*(efield - intEfield)^2)),
      omega0 ~ 0 + pdbid:wm,
      m ~ 0 + pdbid:wm,
      intEfield ~ 0 + pdbid,
      nl = TRUE
    ),
    data = dat,
    iter = 20000,
    chains = 10,
    prior = prior,
    control = list(
      max_treedepth = 15,
      adapt_delta = 0.90
    ),
    cores = detectCores()/2,  # physical cores without hyperthreading
    sample_prior = TRUE
  )


# SERIALISE RESULTS ============================================================

# Reformat Posterior Samples ---------------------------------------------------

post_samples_m <-
  post_samples %>%
  filter(
    is.na(m) == FALSE
  ) %>%
  select(
    chain,
    iter,
    wm,
    pdbid,
    m
  )

post_samples_omega0 <-
  post_samples %>%
  filter(
    is.na(omega0) == FALSE
  ) %>%
  select(
    chain,
    iter,
    wm,
    pdbid,
    omega0
  )

post_samples_int_efield <-
  post_samples %>%
  filter(
    is.na(intEfield) == FALSE
  ) %>%
  select(
    chain,
    iter,
    wm,
    pdbid,
    intEfield
  )

post_samples_sigma <-
  post_samples %>%
  filter(
    is.na(sigma) == FALSE
  ) %>%
  select(
    chain,
    iter,
    wm,
    pdbid,
    sigma
  )

posterior_samples <-
  post_samples_m %>%
  full_join(
    post_samples_int_efield
  ) %>%
  full_join(
    post_samples_omega0
  )


# Export Data to JSON ----------------------------------------------------------

write(
  toJSON(post_samples),
  "posterior_samples.json"
)

write(
  toJSON(me),
  "marginal_effects.json"
)
