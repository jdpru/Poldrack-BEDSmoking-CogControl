# fMRI Self-Regulation Analysis

**Note:** This repository was created temporarily to support my application to Professor Bergelson's lab. The code is normally stored on private servers in the Poldrack Lab until publication but has been uploaded here for demonstration purposes.

## Overview

This repository contains code for first-level modeling of fMRI data, including the creation of design matrices and general linear models (GLMs).

The study investigates **self-regulation** in two exemplar populations with known self-regulatory challenges:

* **Smokers**
* **Individuals with Binge Eating Disorder (BED)**

We employed a **pre/post fMRI scan design**, separated by a four-week behavioral intervention delivered through *Laddr*—a mobile platform for ecological momentary assessment (EMA) and digital self-regulation training.

## Study Design

* **Participants:** 73 total

  * 53 with binge eating disorder
  * 20 who smoke tobacco cigarettes
* **Location:** Stanford University, Palo Alto, CA
* **Timeline:** Participants completed both pre- and post-intervention scans
* **Tasks:**
  Each scan session included the same five fMRI tasks:

  1. Stop-Signal Task [(Logan & Cowan, 1984)](https://doi.org/10.1037/0096-3445.113.4.611)
  2. Motor Selective Stop Task [(De Jong et al., 1995)](https://doi.org/10.1037/0096-3445.124.2.173)
  3. Delay Discounting Task [(Green & Myerson, 2004)](https://doi.org/10.1037/0033-2909.130.5.769)
  4. Cue Reactivity Task with food or tobacco stimuli [(Kober et al., 2010)](https://doi.org/10.1038/nrn2857)
  5. Passive movie viewing task

## Intervention Features

* **BED participants** wore FitBit health trackers to monitor physical activity.
* **Smokers** completed daily mobile breathalyzer assessments (CO ppm).
* The behavioral intervention spanned four weeks and was conducted entirely via the *Laddr* platform.

## Aims

The project seeks to:

* Assess changes in behavioral and neural self-regulation markers post-intervention
* Evaluate how these changes correlate with intervention engagement and outcomes
* Explore individual differences in response to self-regulation training

This research builds on prior findings in cognitive control and self-regulation [(Eisenberg et al., 2019, *Nature Communications*)](https://doi.org/10.1038/s41467-019-09818-8).
