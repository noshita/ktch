# Contributing Documentation

This guide describes the structure, conventions, and writing guidelines　for the ktch documentation.
For build instructions, see `README.md` in　this directory.

## Overview

- URL: <https://doc.ktch.dev/>
- Framework: Sphinx with pydata-sphinx-theme
- Format: MyST Markdown (with myst-nb for executable notebooks)

## Diataxis Structure

The documentation follows the [Diataxis](https://diataxis.fr/) framework.
Each directory corresponds to one quadrant:

| Diataxis Quadrant | Directory | Content |
|-------------------|-----------|---------|
| Tutorial | `tutorials/` | Executable notebooks for learning |
| How-to | `how-to/` | Short task-focused guides |
| Explanation | `explanation/` | Conceptual and theoretical content |
| Reference | `api/` | Auto-generated API documentation |

### Why these names?

`explanation/` instead of `user_guide/`:

- `user_guide` (as in scikit-learn) tends to become a catch-all for mixed content
- `explanation` provides a natural constraint: only conceptual/theoretical content belongs here
- Successful precedents: matplotlib, Django use Diataxis naming

`tutorials/` instead of `examples/`:

- The notebook content is learning-oriented, not just code examples
- "Tutorial" clearly indicates a step-by-step learning experience

## Content Guidelines

### Tutorials (`tutorials/`)

Teach through hands-on experience.

- Step-by-step, explain each action
- Start with a working example
- Build understanding progressively
- Include visualizations

#### Example structure

```markdown
# Tutorial: Elliptic Fourier Analysis

In this tutorial, you will learn to:
- Load outline data
- Compute EFA coefficients
- Visualize results with PCA

## Prerequisites
...

## Step 1: Load the data
...
```

### How-to Guides (`how-to/`)

Help accomplish specific tasks.

- Goal-focused, minimal explanation
- State prerequisites upfront
- Provide copy-paste code
- Link to Explanation for theory

#### Example structure

```markdown
# How to load TPS files

This guide shows how to load landmark data from TPS files.

## Prerequisites
- ktch installed
- A TPS file

## Steps

from ktch.io import read_tps
coords = read_tps("landmarks.tps")

## See also
- {doc}`../explanation/landmark` for GPA theory
- {doc}`../tutorials/landmark/gpa_from_tps` for complete tutorial
```

### Explanation (`explanation/`)

Help understand concepts and theory.

- Discursive, provide context
- Explain trade-offs and alternatives
- Connect to broader scientific context
- Can include mathematical background

#### Example structure

```markdown
# Generalized Procrustes Analysis

Generalized Procrustes Analysis (GPA) is a method for...

## Theoretical Background

The Procrustes problem involves finding the optimal...

## When to Use GPA

GPA is appropriate when...

## Limitations

...

## References
```

### Reference (`api/`)

Provide accurate, complete technical information.

- Auto-generated from docstrings
- Consistent format
- Keep synchronized with code
- Brief descriptions, complete signatures

## Naming Conventions

### Explanation files

Use topic-based naming (not module-prefixed).
The topic name should be self-descriptive and indicate its domain.

| Guideline | Good | Bad |
|-----------|------|-----|
| Use domain-specific terms | `semilandmarks.md` | `landmark_semilandmarks.md` |
| Avoid generic names | `efa_normalization.md` | `normalization.md` |
| Let the topic name indicate domain | `theoretical_shell_models.md` | `models.md` |

#### Rationale

- Explanation documents are organized by concept, not by module
- Users search for "semilandmarks" or "TPS", not "landmark module advanced topics"
- `index.md` provides module-based grouping via sections

#### Example `explanation/index.md`

```markdown
## Landmark-based Morphometrics
- {doc}`landmark` - GPA and shape coordinates
- {doc}`semilandmarks` - Sliding landmarks on curves and surfaces

## Harmonic-based Morphometrics
- {doc}`harmonic` - EFA and spherical harmonics
```

This approach scales to future modules without requiring subdirectories.

## Source File Format

Tutorials are maintained as MyST Markdown directly in `doc/tutorials/`.
myst-nb processes `.md` files with jupytext headers directly, so no copy
or conversion step is needed.

Benefits:

- Single source of truth (no duplication)
- Version control friendly format
- Direct execution by myst-nb during build

## Cross-referencing

Use MyST cross-references to link between sections:

```markdown
For the theory behind this method, see {doc}`../explanation/landmark`.
For step-by-step learning, see {doc}`../tutorials/landmark/generalized_Procrustes_analysis`.
```

Link between related docs across quadrants wherever possible. Do not mix
documentation types in one document.

## External Tool Coverage

When documenting preprocessing or external tool integration:

- GUI tools (ImageJ, tpsDig2, etc.): reference only, link to official documentation
- Python ecosystem tools (OpenCV, scikit-image, etc.): minimal inclusion is OK in tutorials
- Focus on "connection points" — data formats expected by ktch
