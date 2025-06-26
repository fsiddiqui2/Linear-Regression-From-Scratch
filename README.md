# Linear Regression from Scratch in NumPy

This project implements **Linear Regression** using two different approaches:

- **Gradient Descent**: A numerical optimization method for minimizing Mean Squared Error (MSE).
- **Ordinary Least Squares (OLS)**: An analytical solution for finding the line of best fit.

The project is written in Python and uses only core libraries like `NumPy`, `Pandas`, and `Matplotlib`.

## Project Structure

```bash
.
├── linreg_gd.py              # Linear Regression using Gradient Descent
├── linreg_ols.py             # Linear Regression using Ordinary Least Squares
├── util.py                   # Utility functions for MSE, R², etc.
├── test_data/                # Folder containing sample CSV datasets
├── images/                   # Folder to store plots and animation output
├── requirements_pip.txt      # Pip requirements file
├── requirements_conda_mac.txt# Conda requirements file (macOS/arm64)
└── README.md                 # This file
````

## Installation

Create a virtual environment and install dependencies:

### With pip:

```bash
pip install -r requirements_pip.txt
```

### With conda (macOS ARM):

```bash
conda create --name linreg_env --file requirements_conda_mac.txt
conda activate linreg_env
```

## Usage

### 1. Linear Regression via Gradient Descent

```bash
python linreg_gd.py \
    --data_path test_data/data.csv \
    --lr 0.01 \
    --max_iter 10000 \
    --tolerance 1e-6 \
    --graph_name line_gd.png \
    --lossplot_name losses_gd.png \
    --animation_name animation_gd.gif
```

This script will:

* Train a linear model using Gradient Descent
* Plot the loss curve over iterations
* Plot the data with the regression line (if only 1 feature)
* Create an animation of the learning process (if only 1 feature)

### 2. Linear Regression via OLS

```bash
python linreg_ols.py \
    --data_path test_data/data.csv \
    --graph_name line_ols.png
```

This script will:

* Fit a line using the closed-form solution
* Plot the regression result

## Features

* Feature scaling (standardization) for Gradient Descent
* Automatic convergence detection using loss tolerance
* Visualization:

  * Loss curve (`losses_gd.png`)
  * Regression line plot (`line_gd.png` / `line_ols.png`)
  * Training animation (`animation_gd.gif`)
* Implements regression metrics:

  * **MSE (Mean Squared Error)**
  * **R² Score**
  * **RSS / TSS**

## Data Format

CSV input should be structured as:

```
feature1, feature2, ..., target
val1,     val2,     ..., valY
...
```

For OLS, only **one feature** is supported.

## Example Output

```text
Best Fit: y = 2.3210 + (0.4213)x0
MSE: 3.254
R^2: 0.8912
```

Generated files:

* `images/line_gd.png`
* `images/losses_gd.png`
* `images/animation_gd.gif`

## TODOs

* [ ] Support multivariate OLS
* [ ] Add test suite
* [ ] Add command-line option to disable scaling

## 📄 License

MIT License. Feel free to use and modify.
