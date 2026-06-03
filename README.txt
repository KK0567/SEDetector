# SEDetector

SEDetector is a semantic evidence-driven framework for APT attack detection under privacy-constrained environments. The framework transforms raw host logs and network observations into irreversible behavior-level semantic representations and constructs semantic hypergraphs for threat detection and evidence-level interpretation.

Instead of directly modeling raw logs or traffic records, SEDetector uses semantic evidence units as the intermediate representation. This design reduces the dependence on sensitive raw observations while preserving security-relevant behavioral semantics for downstream detection.

## Overview

SEDetector is built around three core components:

1. **Irreversible behavior-level semantic abstraction**  
   Raw security observations are transformed into abstract semantic evidence units. The abstraction removes sensitive contextual details while retaining detection-related behavioral semantics.

2. **Semantic hypergraph-based threat modeling**  
   Semantic evidence units are organized into a semantic hypergraph to capture high-order behavioral dependencies among entities, stages, techniques, and contextual evidence.

3. **Evidence-level explainability**  
   The framework provides evidence-level interpretation for detection decisions by identifying semantic evidence units and semantic tokens that contribute to the prediction. Hyperedge-level information is used as contextual support rather than the primary explanation granularity.

## Supported Datasets

The repository contains experimental materials for multiple APT-related datasets:

- OpTC
- TCE5
- DAPT

These datasets are used to evaluate SEDetector under different data modalities and attack-stage settings.

## Environment

Install the required dependencies with:

```bash
pip install -r requirements.txt
