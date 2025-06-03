#!/bin/bash
# Script to run categorize_in_sverdrup.py

# Define the Python script name
PYTHON_SCRIPT="categorize.py"

# Optionally log output to a file
LOG_FILE="categorize.log"

echo "Starting the categorization in Sverdrup units..."

# Run the Python script and log output
python3 "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1

# Check exit status
if [ $? -eq 0 ]; then
    echo "Categorization completed successfully."
else
    echo "Error during categorization. Check the log file: $LOG_FILE"
fi

