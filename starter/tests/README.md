# Local Testing with run_tests.py

This directory contains tests for the project. The `run_tests.py` script in the parent directory can be used to run these tests locally, which helps identify and fix issues that might cause GitHub Actions to fail.

## Running the Tests

To run the tests, navigate to the `starter` directory and run:

```bash
python run_tests.py
```

This will:
1. Check if pytest and flake8 are installed, and install them if they're not
2. Run flake8 to check for critical code style issues
3. Run pytest to execute the tests

## Command-Line Arguments

The `run_tests.py` script supports several command-line arguments:

- `--flake8-only`: Run only flake8 checks
  ```bash
  python run_tests.py --flake8-only
  ```

- `--pytest-only`: Run only pytest tests
  ```bash
  python run_tests.py --pytest-only
  ```

- `--critical-only`: Check only for critical flake8 issues (ignores style warnings)
  ```bash
  python run_tests.py --critical-only
  ```

- `--test-path`: Specify a specific path to test
  ```bash
  python run_tests.py --test-path tests/test_data.py
  ```

## Test Files

The tests in this directory include:

- `test_data.py`: Tests for the data processing functionality in `starter/ml/data.py`
  - `test_process_data_without_label`: Tests the `process_data` function without providing a label
  - `test_process_data_with_label`: Tests the `process_data` function with a label
  - `test_process_data_inference`: Tests the `process_data` function in inference mode

## Adding New Tests

To add new tests:

1. Create a new file with a name starting with `test_` (e.g., `test_model.py`)
2. Write test functions with names starting with `test_`
3. Run the tests using `run_tests.py`

## GitHub Actions

The GitHub Actions workflow in `.github/workflows/python-test.yml` runs the same tests as `run_tests.py`. If the tests pass locally, they should also pass in the GitHub Actions workflow.