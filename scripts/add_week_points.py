from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd


def _read_points_lines(points_path: Path | None) -> list[int]:
    if points_path is None:
        raw = sys.stdin.read()
        source = "stdin"
    else:
        raw = points_path.read_text(encoding="utf-8")
        source = str(points_path)

    lines: list[str] = [line.strip() for line in raw.splitlines()]
    lines = [line for line in lines if line]

    if not lines:
        raise SystemExit(
            f"No points provided from {source}. Provide one integer per line via stdin or --points-file."
        )

    points: list[int] = []
    for idx, line in enumerate(lines, start=1):
        try:
            points.append(int(line))
        except ValueError as exc:
            raise SystemExit(
                f"Invalid points value on line {idx} from {source}: {line!r} (expected integer)"
            ) from exc

    return points


def _read_text_input(path: Path | None, *, empty_error: str) -> tuple[str, str]:
    if path is None:
        raw = sys.stdin.read()
        source = "stdin"
    else:
        raw = path.read_text(encoding="utf-8")
        source = str(path)

    if not raw.strip():
        raise SystemExit(empty_error)

    return raw, source


def _load_points_csv(points_csv: Path) -> pd.DataFrame:
    if not points_csv.exists():
        raise SystemExit(f"Points file not found: {points_csv}")

    df = pd.read_csv(points_csv)
    required = {"manager", "season", "gw", "gw_points"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(
            f"{points_csv} is missing required columns: {', '.join(sorted(missing))}"
        )

    df["manager"] = df["manager"].astype(str).str.strip()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["gw"] = pd.to_numeric(df["gw"], errors="raise").astype(int)
    df["gw_points"] = pd.to_numeric(df["gw_points"], errors="raise").astype(int)

    return df


def _season_managers(df: pd.DataFrame, season: int) -> list[str]:
    managers = sorted(
        df.loc[df["season"] == season, "manager"].dropna().unique().tolist(),
        key=lambda s: str(s).lower(),
    )
    if not managers:
        raise SystemExit(f"No managers found for season {season} in points.csv")
    return managers


def _normalize_handle(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _replace_or_insert_gw_block(
    df: pd.DataFrame, *, season: int, gw: int, new_rows: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Replace the (season, gw) block while preserving all other row ordering.

    - Existing rows keep their relative order.
    - The new GW block is inserted where the old block started, or at a season-appropriate
      location if it didn't exist.
    """

    if set(new_rows.columns) != {"manager", "season", "gw", "gw_points"}:
        raise SystemExit("Internal error: new_rows has unexpected columns")

    mask_target = (df["season"] == season) & (df["gw"] == gw)
    existing_idx = df.index[mask_target].tolist()

    season_mask = df["season"] == season
    season_idx = df.index[season_mask].tolist()

    if existing_idx:
        insert_pos = min(existing_idx)
    else:
        if not season_idx:
            insert_pos = len(df)
        else:
            # Insert after the last row for the largest GW < target GW within this season.
            season_rows = df.loc[season_mask, ["gw"]]
            prior_gws = sorted({int(x) for x in season_rows["gw"].unique().tolist() if int(x) < gw})
            if prior_gws:
                prev_gw = prior_gws[-1]
                prev_idx = df.index[(df["season"] == season) & (df["gw"] == prev_gw)].max()
                insert_pos = int(prev_idx) + 1
            else:
                insert_pos = min(season_idx)

    # Remove existing target rows.
    removed_before = 0
    if existing_idx:
        removed_before = sum(1 for i in existing_idx if i < insert_pos)
    insert_pos -= removed_before

    df_kept = df.loc[~mask_target]
    records = df_kept.to_dict("records")
    new_records = new_rows.to_dict("records")
    records[insert_pos:insert_pos] = new_records
    updated = pd.DataFrame.from_records(records, columns=["manager", "season", "gw", "gw_points"])

    action = "overwrote" if existing_idx else "inserted"
    return updated, action


def _parse_google_sheet_points_table(
    raw: str, *, season_managers: list[str]
) -> tuple[dict[str, dict[int, int]], list[int]]:
    """Parse TSV-like text copied from Google Sheets.

    Expected: one header row containing GW columns like GW1..GW38, and one row per player.
    Returns: mapping manager_handle -> {gw: points} and sorted list of gws present (>=1).
    """

    norm_to_manager = {_normalize_handle(m): m for m in season_managers}
    expected_norms = set(norm_to_manager.keys())

    rows = list(csv.reader(raw.splitlines(), delimiter="\t"))
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        raise SystemExit("No rows found in sheet input")

    header = [cell.strip() for cell in rows[0]]
    gw_cols: list[tuple[int, int]] = []
    for idx, col in enumerate(header):
        m = re.match(r"^GW\s*(\d+)$", col.strip(), flags=re.IGNORECASE)
        if not m:
            continue
        gw_num = int(m.group(1))
        if gw_num <= 0:
            continue
        gw_cols.append((idx, gw_num))

    if not gw_cols:
        raise SystemExit("No GW columns found (expected headers like GW1, GW2, ...)")

    gw_cols = sorted(gw_cols, key=lambda t: t[1])
    gws = [gw for _, gw in gw_cols]

    by_manager: dict[str, dict[int, int]] = {}
    for row_idx, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        player = row[0].strip()
        if not player:
            continue

        norm = _normalize_handle(player)
        if norm not in norm_to_manager:
            raise SystemExit(
                f"Unknown player on sheet row {row_idx}: {player!r}. "
                "Player names must map to manager handles in points.csv (e.g. 'Charlie BN' -> 'charliebn')."
            )
        manager = norm_to_manager[norm]
        if manager in by_manager:
            raise SystemExit(f"Duplicate player row for {player!r} (maps to manager {manager!r})")

        gw_points: dict[int, int] = {}
        for col_idx, gw in gw_cols:
            cell = row[col_idx].strip() if col_idx < len(row) else ""
            if cell == "":
                raise SystemExit(
                    f"Missing points value for player {player!r} at GW{gw} (sheet row {row_idx})."
                )
            try:
                gw_points[gw] = int(cell)
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid points value for player {player!r} at GW{gw}: {cell!r} (expected integer)"
                ) from exc

        by_manager[manager] = gw_points

    seen_norms = {_normalize_handle(m) for m in by_manager.keys()}
    missing = sorted(expected_norms - seen_norms)
    extra = sorted(seen_norms - expected_norms)
    if missing or extra:
        missing_handles = [norm_to_manager[n] for n in missing if n in norm_to_manager]
        raise SystemExit(
            "Sheet must contain exactly the season managers from points.csv. "
            f"Missing: {missing_handles}. Extra(normalized): {extra}."
        )

    return by_manager, gws


def _latest_gw_is_complete(season_df: pd.DataFrame, expected_managers: set[str], gw: int) -> bool:
    gw_managers = set(
        season_df.loc[season_df["gw"] == gw, "manager"].dropna().astype(str).tolist()
    )
    return gw_managers == expected_managers


def _choose_season_and_gw(
    df: pd.DataFrame, season_arg: int | None, gw_arg: int | None
) -> tuple[int, int, bool]:
    if season_arg is None:
        season = int(df["season"].max())
    else:
        season = int(season_arg)
        if (df["season"] == season).sum() == 0:
            raise SystemExit(f"Season {season} not found in points.csv")

    season_df = df.loc[df["season"] == season]
    expected = set(_season_managers(df, season))

    if gw_arg is not None:
        gw = int(gw_arg)
        if not (1 <= gw <= 38):
            raise SystemExit(f"--gw must be between 1 and 38 (got {gw})")
        return season, gw, True

    latest_gw = int(season_df["gw"].max())
    if not _latest_gw_is_complete(season_df, expected, latest_gw):
        raise SystemExit(
            "Latest GW appears partially filled in points.csv; specify --gw to overwrite that GW with complete data."
        )

    gw = latest_gw + 1
    if gw > 38:
        raise SystemExit(
            f"Season {season} already has complete data through GW {latest_gw}; refusing to default to GW {gw}. "
            "Specify --season/--gw explicitly if you are backfilling."
        )

    return season, gw, False


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp_path = Path(tmp.name)
        df.to_csv(tmp, index=False)

    tmp_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Append or overwrite a single GW in raw/points.csv. "
            "Points are provided as one integer per line, in manager alphabetical order (by lowercase handle)."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["append-latest", "validate-season"],
        default="append-latest",
        help="Operation mode (use --validate-season as a convenience flag).",
    )

    parser.add_argument(
        "--validate-season",
        action="store_true",
        help="Validate/correct points.csv for a season using a Google Sheets table from stdin or --sheet-file.",
    )

    parser.add_argument(
        "--points-csv",
        type=Path,
        default=Path("raw/points.csv"),
        help="Path to points.csv (default: raw/points.csv)",
    )

    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season (e.g., 2025). Defaults to max season in points.csv.",
    )

    parser.add_argument(
        "--gw",
        type=int,
        default=None,
        help=(
            "Gameweek to write. If omitted, defaults to (max gw + 1) for the latest season, but errors if the latest GW is partially filled."
        ),
    )

    parser.add_argument(
        "--points-file",
        type=Path,
        default=None,
        help="For append mode: file containing one integer points value per line. If omitted, read from stdin.",
    )

    parser.add_argument(
        "--sheet-file",
        type=Path,
        default=None,
        help="For --validate-season: TSV copied from Google Sheets. If omitted, read from stdin.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing points.csv.",
    )

    args = parser.parse_args(argv)

    if args.validate_season:
        args.mode = "validate-season"

    df = _load_points_csv(args.points_csv)

    if args.mode == "validate-season":
        season = int(df["season"].max()) if args.season is None else int(args.season)
        if (df["season"] == season).sum() == 0:
            raise SystemExit(f"Season {season} not found in points.csv")

        managers = _season_managers(df, season)
        raw, source = _read_text_input(
            args.sheet_file,
            empty_error="No sheet data provided. Paste the Google Sheets table into stdin or use --sheet-file.",
        )

        print(
            f"Validating season={season} in {args.points_csv} using sheet input from {source}...",
            file=sys.stderr,
        )

        sheet_by_manager, gws = _parse_google_sheet_points_table(raw, season_managers=managers)

        # Build a fast lookup of existing points, ensuring uniqueness.
        season_df = df.loc[df["season"] == season, ["manager", "gw", "gw_points"]]
        dupes = season_df.duplicated(subset=["manager", "gw"], keep=False)
        if dupes.any():
            bad = season_df.loc[dupes].sort_values(["manager", "gw"]).head(20)
            raise SystemExit(
                "Duplicate (manager, gw) rows found for the season in points.csv; refusing to validate. "
                f"Examples:\n{bad.to_string(index=False)}"
            )

        existing_lookup = {
            (str(r.manager), int(r.gw)): int(r.gw_points)
            for r in season_df.itertuples(index=False)
        }

        planned_updates: list[tuple[int, str, int | None, int]] = []
        for gw in gws:
            for manager in managers:
                new_val = int(sheet_by_manager[manager][gw])
                old_val = existing_lookup.get((manager, gw))
                if old_val is None or int(old_val) != new_val:
                    planned_updates.append((gw, manager, old_val, new_val))

        if not planned_updates:
            print("No changes needed.")
            return 0

        # Output planned changes before applying.
        print(f"Planned changes for season {season}:")
        by_gw: dict[int, list[tuple[str, int | None, int]]] = {}
        for gw, manager, old_val, new_val in planned_updates:
            by_gw.setdefault(gw, []).append((manager, old_val, new_val))

        for gw in sorted(by_gw):
            changes = by_gw[gw]
            print(f"GW{gw}:")
            for manager, old_val, new_val in changes:
                if old_val is None:
                    print(f"  {manager}: (missing) -> {new_val}")
                else:
                    print(f"  {manager}: {old_val} -> {new_val}")

        if args.dry_run:
            return 0

        updated_df = df
        for gw in gws:
            block = pd.DataFrame(
                {
                    "manager": managers,
                    "season": [season] * len(managers),
                    "gw": [gw] * len(managers),
                    "gw_points": [int(sheet_by_manager[m][gw]) for m in managers],
                }
            )
            updated_df, _ = _replace_or_insert_gw_block(updated_df, season=season, gw=gw, new_rows=block)

        # Verify no partials remain for provided gws.
        for gw in gws:
            gw_df = updated_df.loc[(updated_df["season"] == season) & (updated_df["gw"] == gw)]
            gw_managers = set(gw_df["manager"].astype(str).tolist())
            if gw_managers != set(managers):
                raise SystemExit(
                    f"Internal error: after applying changes, season {season} GW{gw} is incomplete."
                )

        _atomic_write_csv(updated_df[["manager", "season", "gw", "gw_points"]], args.points_csv)
        print(f"Applied {len(planned_updates)} changes to {args.points_csv}")
        return 0

    season, gw, gw_was_explicit = _choose_season_and_gw(df, args.season, args.gw)

    managers = _season_managers(df, season)
    input_source = "stdin" if args.points_file is None else str(args.points_file)
    print(
        f"About to write season={season} gw={gw} in {args.points_csv}. "
        f"Expecting {len(managers)} points values (one per line) from {input_source}.",
        file=sys.stderr,
    )
    points = _read_points_lines(args.points_file)

    if len(points) != len(managers):
        raise SystemExit(
            f"Expected {len(managers)} points values (one per manager in season {season}), got {len(points)}. "
            f"Manager order is: {', '.join(managers)}"
        )

    season_df = df.loc[df["season"] == season]
    existing_gw_df = season_df.loc[season_df["gw"] == gw]
    existing_managers = set(existing_gw_df["manager"].astype(str).tolist())
    expected_managers = set(managers)

    if existing_managers and not gw_was_explicit:
        raise SystemExit(
            f"GW {gw} in season {season} already has data; specify --gw explicitly to overwrite."
        )

    unexpected = existing_managers - expected_managers
    if unexpected:
        raise SystemExit(
            f"Unexpected managers already present for season {season} gw {gw}: {', '.join(sorted(unexpected))}. "
            "Refusing to overwrite."
        )

    new_rows = pd.DataFrame(
        {
            "manager": managers,
            "season": [season] * len(managers),
            "gw": [gw] * len(managers),
            "gw_points": points,
        }
    )

    updated, _ = _replace_or_insert_gw_block(df, season=season, gw=gw, new_rows=new_rows)

    # Ensure no partial data remains.
    final_gw_df = updated.loc[(updated["season"] == season) & (updated["gw"] == gw)]
    final_managers = set(final_gw_df["manager"].astype(str).tolist())
    if final_managers != expected_managers:
        missing = sorted(expected_managers - final_managers)
        extra = sorted(final_managers - expected_managers)
        raise SystemExit(
            "Refusing to write partial GW data. "
            f"Missing managers: {missing}. Extra managers: {extra}."
        )

    # Keep original ordering of existing rows; ensure standard column ordering.
    updated = updated[["manager", "season", "gw", "gw_points"]]

    if args.dry_run:
        action = "Overwrite" if existing_managers else "Append"
        print(f"{action}: season={season}, gw={gw}")
        for manager, pts in zip(managers, points, strict=True):
            print(f"  {manager}: {pts}")
        return 0

    _atomic_write_csv(updated, args.points_csv)

    action = "Overwrote" if existing_managers else "Appended"
    print(f"{action} season {season} GW {gw} to {args.points_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

