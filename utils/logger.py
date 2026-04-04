"""
Logger Utility
==============
Handles logging to both console and file throughout the pipeline.
"""

import os
import datetime


class PipelineLogger:
    def __init__(self, log_file, verbose=True):
        self.log_file = log_file
        self.verbose = verbose
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Initialize log file
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"Pipeline Log - Started at {datetime.datetime.now()}\n")
            f.write("=" * 60 + "\n\n")

    def log(self, message, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
        # Print to console if verbose
        if self.verbose:
            print(formatted)

    def section(self, title):
        separator = "=" * 60
        self.log(separator)
        self.log(f"  {title}")
        self.log(separator)

    def success(self, message):
        self.log(message, level="SUCCESS")

    def warning(self, message):
        self.log(message, level="WARNING")

    def error(self, message):
        self.log(message, level="ERROR")