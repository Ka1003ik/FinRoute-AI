import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cfpb_2026_narratives.csv"
REPORT_PATH = ROOT / "outputs" / "reports" / "duplicate_analysis.json"

TEXT = "Consumer complaint narrative"
LABEL = "Product"


def main() -> None:
    df = pd.read_csv(
        DATA_PATH,
        usecols=[TEXT, LABEL],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    df[TEXT] = df[TEXT].str.strip()
    df[LABEL] = df[LABEL].str.strip()

    grouped = df.groupby(TEXT, sort=False)[LABEL].agg(
        rows="size",
        product_count="nunique",
    )
    repeated = grouped[grouped["rows"] > 1]
    same_product = repeated[repeated["product_count"] == 1]
    conflicting = repeated[repeated["product_count"] > 1]

    examples = []
    for narrative in conflicting.index[:10]:
        labels = sorted(df.loc[df[TEXT] == narrative, LABEL].unique())
        examples.append({
            "rows": int(grouped.loc[narrative, "rows"]),
            "products": labels,
        })

    report = {
        "total_rows": int(len(df)),
        "repeated_narrative_groups": int(len(repeated)),
        "extra_rows_from_repeats": int((repeated["rows"] - 1).sum()),
        "groups_repeated_with_same_product": int(len(same_product)),
        "groups_repeated_with_conflicting_products": int(len(conflicting)),
        "extra_rows_in_conflicting_groups": int(
            (conflicting["rows"] - 1).sum()
        ),
        "conflict_examples_without_narrative_text": examples,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()