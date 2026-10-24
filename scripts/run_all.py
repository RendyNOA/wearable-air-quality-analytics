"""Run all four analyses on data/demo (default) or your own prepared folder."""
import argparse
from pathlib import Path

from scripts import calibration_analysis, har_analysis, movement_analysis, sampling_analysis


def main():
    parser = argparse.ArgumentParser(description="Wearable air-quality sensor analytics")
    parser.add_argument("--data-dir", default="data/demo", help="Folder with the four input CSV files")
    parser.add_argument("--out-dir", default="outputs/demo", help="Where tables and figures are written")
    args = parser.parse_args()
    steps = (("1/4 Calibration", calibration_analysis),
             ("2/4 Static vs dynamic", movement_analysis),
             ("3/4 Activity recognition", har_analysis),
             ("4/4 Adaptive sampling", sampling_analysis))
    for title, module in steps:
        print(f"\n== {title} ==")
        module.run(Path(args.data_dir), Path(args.out_dir))
    print(f"\nDone. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
