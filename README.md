# Linear Regression CLI Tool

This is a simple Python command-line tool that fits a **simple linear regression model** to a given list of `x` and `y` values and outputs the **line of best fit** and **R² score**. This linear regression is implemented using only **numpy**.

## Features

- Computes the least squares regression line:  
  \[
  y = b_0 + b_1x
  \]
- Prints model parameters (`b₀`, `b₁`)
- Calculates the coefficient of determination (**R²**)
- Supports command-line arguments for easy use

## Requirements

- Python 3.10+
- NumPy

Install dependencies (if needed):

```bash
pip install numpy
````

## Usage

### Run from the command line:

```bash
python linear_regression.py --x 1 2 3 4 5 --y 2 4 5 4 5
```

### Output:

```
Line of Best Fit: y = 2.2000 + 0.6000x
R^2: 0.4800
```

## How it Works

The script:

1. Parses `--x` and `--y` values using `argparse`
2. Computes:

   * Means of `x` and `y`
   * Slope `b₁` and intercept `b₀`
   * R² score (model accuracy)
3. Predicts values using:

   $$
   \hat{y} = b_0 + b_1x
   $$

---

## File Structure

```
linear_regression.py      # Main script containing the LinearModel class
```

---

## Example

```bash
python linear_regression.py --x 1 2 3 --y 2 4 6
```

Output:

```
Line of Best Fit: y = 0.0000 + 2.0000x
R^2: 1.0
```

---

## License

This project is licensed under the MIT License.

---

## Author

Farhaan Siddiqui
