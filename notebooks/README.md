# Notebooks Directory

Jupyter notebooks for exploratory analysis, prototyping, and visualization.

## Purpose

This directory contains notebooks for:
- Data exploration and analysis
- Strategy prototyping and testing
- Model training and evaluation
- Visualization and reporting
- Documentation and examples

## Naming Convention

Use descriptive names with prefixes:
- `01_exploratory_*.ipynb` - Exploratory data analysis
- `02_feature_*.ipynb` - Feature engineering
- `03_model_*.ipynb` - Model development
- `04_strategy_*.ipynb` - Strategy prototyping
- `05_analysis_*.ipynb` - Results analysis

## Best Practices

1. **Clear documentation**: Add markdown cells explaining each step
2. **Reproducibility**: Set random seeds, document versions
3. **Clean up**: Remove unused cells before committing
4. **Export results**: Save figures and key results to files
5. **Version control**: Clear outputs before committing (optional)

## Usage

Run notebooks using Jupyter:

```bash
# Start Jupyter Lab
uv run jupyter lab

# Or Jupyter Notebook
uv run jupyter notebook
```

## Integration with Source Code

Prototype in notebooks, then move production code to `src/`:
1. Develop and test in notebook
2. Refactor into clean functions
3. Move to appropriate `src/` module
4. Add unit tests in `tests/`
5. Keep notebook as documentation/example
