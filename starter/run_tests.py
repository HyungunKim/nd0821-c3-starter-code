#!/usr/bin/env python
"""
Script to run flake8 and pytest locally.
This helps identify and fix issues that might cause GitHub Actions to fail.
"""

import os
import sys
import subprocess
import argparse


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import pytest
        print("pytest is installed.")
    except ImportError:
        print("pytest is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        print("pytest installed successfully.")

    try:
        import flake8
        print("flake8 is installed.")
    except ImportError:
        print("flake8 is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flake8"])
        print("flake8 installed successfully.")


def run_flake8(args):
    """Run flake8 with the same settings as the GitHub action."""
    print("\n=== Running flake8 ===")
    
    # First check for critical errors (same as GitHub action)
    critical_result = subprocess.run(
        ["flake8", ".", "--count", "--select=E9,F63,F7,F82", "--show-source", "--statistics"],
        capture_output=True,
        text=True
    )
    
    print(critical_result.stdout)
    if critical_result.stderr:
        print("Errors:", critical_result.stderr)
    
    if critical_result.returncode != 0:
        print("Critical flake8 issues found. These will cause the GitHub action to fail.")
    else:
        print("No critical flake8 issues found.")
    
    # Then check for warnings (exit-zero means it won't fail the build)
    if not args.critical_only:
        print("\n--- Checking for style warnings ---")
        warning_result = subprocess.run(
            ["flake8", ".", "--count", "--exit-zero", "--max-complexity=10", 
             "--max-line-length=127", "--statistics"],
            capture_output=True,
            text=True
        )
        
        print(warning_result.stdout)
        if warning_result.stderr:
            print("Errors:", warning_result.stderr)
        
        if warning_result.stdout:
            print("Style warnings found. These won't cause the GitHub action to fail, but fixing them is recommended.")
        else:
            print("No style warnings found.")
    
    return critical_result.returncode


def run_pytest(args):
    """Run pytest to execute tests."""
    print("\n=== Running pytest ===")
    
    # Check if there are any test files
    import glob
    test_files = glob.glob("**/test_*.py", recursive=True)
    
    if not test_files:
        print("No test files found. Create test files with names starting with 'test_'.")
        print("Creating a sample test file for demonstration...")
        
        # Create a tests directory if it doesn't exist
        os.makedirs("tests", exist_ok=True)
        
        # Create a simple test file
        with open("tests/test_sample.py", "w") as f:
            f.write("""
def test_sample():
    \"\"\"A sample test that always passes.\"\"\"
    assert True
""")
        print("Created a sample test file at tests/test_sample.py")
        test_files = ["tests/test_sample.py"]
    
    # Run pytest
    pytest_args = ["-v"]
    if args.test_path:
        pytest_args.append(args.test_path)
    
    print(f"Running tests in: {test_files if not args.test_path else args.test_path}")
    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + pytest_args,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    if result.returncode != 0:
        print("Some tests failed. Fix the failing tests to make the GitHub action pass.")
    else:
        print("All tests passed!")
    
    return result.returncode


def main():
    """Main function to run the tests."""
    parser = argparse.ArgumentParser(description="Run flake8 and pytest locally.")
    parser.add_argument("--flake8-only", action="store_true", help="Run only flake8")
    parser.add_argument("--pytest-only", action="store_true", help="Run only pytest")
    parser.add_argument("--critical-only", action="store_true", 
                        help="Check only for critical flake8 issues")
    parser.add_argument("--test-path", type=str, help="Specific path to test")
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    flake8_status = 0
    pytest_status = 0
    
    # Run flake8 if requested or if no specific tool is requested
    if args.flake8_only or (not args.flake8_only and not args.pytest_only):
        flake8_status = run_flake8(args)
    
    # Run pytest if requested or if no specific tool is requested
    if args.pytest_only or (not args.flake8_only and not args.pytest_only):
        pytest_status = run_pytest(args)
    
    # Return non-zero exit code if any tool failed
    if flake8_status != 0 or pytest_status != 0:
        print("\n=== Summary ===")
        if flake8_status != 0:
            print("❌ flake8 found critical issues.")
        if pytest_status != 0:
            print("❌ pytest found failing tests.")
        sys.exit(1)
    else:
        print("\n=== Summary ===")
        print("✅ All checks passed!")


if __name__ == "__main__":
    main()