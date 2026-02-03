# aiaiai

The Acton Invitational Artificial Intelligence Assistant by Iain

## Update points.csv

Append/overwrite a single gameweek:

    printf "48\n31\n62\n..." | python scripts/add_week_points.py

Validate/correct an entire season from a Google Sheets paste (TSV with columns GW1..GW38, one row per player):

    pbpaste | python scripts/add_week_points.py --validate-season --season 2025 --dry-run
    pbpaste | python scripts/add_week_points.py --validate-season --season 2025

Setup:

    pip install uv

    uv venv
    source .venv/bin/activate
    uv sync
