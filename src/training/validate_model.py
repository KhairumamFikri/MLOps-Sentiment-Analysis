import json
import sys

THRESHOLD = 0.80

with open(
    "metrics.json"
) as f:

    metrics = json.load(f)

f1 = metrics["f1_score"]

print(
    f"F1 Score: {f1}"
)

if f1 < THRESHOLD:

    print(
        "Validation Failed"
    )

    sys.exit(1)

print(
    "Validation Passed"
)