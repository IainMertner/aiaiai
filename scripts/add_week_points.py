from __future__ import annotations

import argparse
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
        help="Operation mode. 'validate-season' is reserved for future implementation.",
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
        help="File containing one integer points value per line. If omitted, read from stdin.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing points.csv.",
    )

    args = parser.parse_args(argv)

    if args.mode == "validate-season":
        print(
            "Mode 'validate-season' is not implemented yet. Use --mode append-latest for now.",
            file=sys.stderr,
        )
        return 2

    df = _load_points_csv(args.points_csv)
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

    # Preserve existing file order: drop any existing rows for this (season, gw)
    # and append the complete GW block at the end.
    keep_mask = ~((df["season"] == season) & (df["gw"] == gw))
    updated = pd.concat([df.loc[keep_mask], new_rows], ignore_index=True)

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

