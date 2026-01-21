<p align="center">
 <h2 align="center"> Which Reasoning Trajectories Teach Students to Reason Better? <br> A Simple Metric of Informative Alignment </h2>
</p>

## 📋 Overview

In this work, we study data–student suitability in reasoning distillation and propose **Rank-Surprisal Ratio** (RSR), a simple metric for identifying suitable reasoning trajectories for a given student.
Motivated by our analysis, RSR jointly captures a trajectory's informativeness and alignment with the student’s behavior, favoring trajectories with low absolute probability but relatively high-ranked tokens.

- 📖 **Paper**: [Read our paper on arXiv](https://arxiv.org/abs/2601.14249)
- 🛠️ **Code**: Available in this repository.

Our codebase supports the computation of our suitability metric, the **Rank-Surprisal Ratio**, given teacher trajectories and student models.

Additional materials, including reasoning trajectory datasets, are planned for future release.

## 🚀 Quick Start

`rsr_launch.py` is the entry-point script for computing RSR. You can modify the global variables in this file according to your experimental setup and then run it to start the computation.  
`rsr_cal.py` contains the core computation logic, with placeholders and comments provided to facilitate easy customization. The code has been cleaned up from our original implementation.

The current implementation depends only on `torch` and `transformers`. We recommend the following environment configuration:

* torch==2.7.0
* transformers==4.53.3
* CUDA 12.8

