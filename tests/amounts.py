"""The numbers a chart takes from speech, checked on fixed transcripts.

    .venv/bin/python tests/amounts.py

Each case is a line as the transcriber writes it (figures, "2K", "$2,000", "15%", or numbers in
words), with every word given a time. The check is what `am.series` takes from it, how each
value is written on the card, the label found in the speaker's words, and that each amount keeps
the time of the word that says it. Nothing is rendered and no model runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK))

from fx.core import amounts as am  # noqa: E402
from fx.core.words import Word  # noqa: E402

STEP = 0.32  # seconds a word in the fixtures


def words_of(line: str) -> list[Word]:
    return [Word(w, round(i * STEP, 3), round(i * STEP + STEP * 0.85, 3)) for i, w in enumerate(line.split())]


# (the line as transcribed, the values drawn oldest first, how each is written, the label, the pattern)
CASES = [
    ("We went from 2K followers to 197K in six months.",
     [2000, 197000], ["2K", "197K"], "followers", "from-to"),
    ("Revenue grew from $2,000 to $15,000 and then to $197,000 this year.",
     [2000, 15000, 197000], ["$2K", "$15K", "$197K"], "Revenue", "from-to"),
    ("In 2024 we hit 1.5 million users, up from 200,000 the year before.",
     [200000, 1500000], ["200K", "1.5M"], "users", "up-from"),
    ("Our churn dropped from 12% to 3.5% in a quarter.",
     [12, 3.5], ["12%", "3.5%"], "churn", "from-to"),
    ("We now have two hundred and fifty thousand views.",
     [250000], ["250K"], "views", ""),
    ("Views jumped to 1.2M, then 3.4M, then 8.9M.",
     [1.2e6, 3.4e6, 8.9e6], ["1.2M", "3.4M", "8.9M"], "Views", ""),
    ("From two to ten customers in a single year.",
     [2, 10], ["2", "10"], "customers", "from-to"),
    ("It takes 3 weeks and costs $500 to start.",
     [500], ["$500"], "", ""),
    ("We went from 2K in 2023 to 197K in 2024.",
     [2000, 197000], ["2K", "197K"], "", "from-to"),
    ("Sales are up one point five million dollars this quarter.",
     [1.5e6], ["$1.5M"], "Sales", ""),
    ("We grew 15 percent to 40 percent in two years.",
     [15, 40], ["15%", "40%"], "", ""),
    ("We are now 10x faster than last year.",
     [10], ["10x"], "faster", ""),
    ("A hundred and ninety-seven K followers now.",
     [197000], ["197K"], "followers", ""),
]
# Lines a chart must not take a number from.
NOTHING = [
    "There are two ways to do this, and one of them is free.",
    "Back in 2019 I started this channel, in twenty twenty-one I quit my job.",
    "Give me five minutes and three seconds.",
    "Meet me at 7 on the second floor.",
]
# What the pack's own transcriber wrote for two read-out lines (28 October 2026): it splits a figure
# into pieces ("2" ",000", "12" "%", "19" ".5%,"), which must be joined back with their times.
REAL = [
    ("So@0.0@0.22 here@0.22@0.42 is@0.42@0.58 the@0.58@0.74 thing.@0.74@1.02 Last@1.4@1.56 year@1.56@1.78 "
     "we@1.78@1.96 went@1.96@2.14 from@2.14@2.36 2@2.36@2.62 ,000@2.62@3.06 followers@3.06@3.44 to@3.44@3.74 "
     "15@3.74@4.14 ,000,@4.14@4.76 and@4.9@5.08 today@5.08@5.34 we@5.34@5.52 are@5.52@5.7 at@5.7@5.82 "
     "197@5.82@6.8 ,000.@6.8@7.86 Our@8.02@8.18 revenue@8.18@8.48 grew@8.48@8.76 from@8.76@9.0 12@9.0@9.32 "
     "%@9.32@9.7 to@9.7@9.92 40@9.92@10.26 %@10.26@10.68 margin.@10.68@11.02 We@11.4@11.54 now@11.54@11.76 "
     "have@11.76@11.98 2@11.98@12.2 .5@12.2@12.82 million@12.82@13.06 views.@13.06@13.5",
     # the runs, each as (values, written, the second each lands, label), as `series` finds them near a time
     [(None, [2000, 15000, 197000], ["2K", "15K", "197K"], [2.36, 3.74, 5.82], "followers"),
      (9.5, [12, 40], ["12%", "40%"], [9.0, 9.92], "margin"),
      (12.5, [2.5e6], ["2.5M"], [11.98], "views")]),
    ("In@0.0@0.16 three@0.16@0.42 months,@0.42@0.84 our@1.16@1.34 followers@1.34@1.76 went@1.76@2.08 "
     "from@2.08@2.32 2k@2.32@2.78 to@2.78@2.98 197k,@2.98@4.68 sales@4.68@4.98 are@4.98@5.26 up@5.26@5.44 "
     "19@5.44@5.94 .5%,@5.94@7.02 we@7.02@7.6 raised@7.6@7.88 3@7.88@8.2 million@8.2@8.42 dollars,@8.42@8.98 "
     "up@9.28@9.44 from@9.44@9.7 500@9.7@10.24 ,000,@10.24@10.94 it@11.28@11.42 is@11.42@11.56 now@11.56@11.8 "
     "10x@11.8@12.26 faster,@12.26@12.8 and@13.12@13.24 churn@13.24@13.46 fell@13.46@13.8 from@13.8@14.06 "
     "12@14.06@14.42 %@14.42@14.8 to@14.8@15.04 3@15.04@15.32 and@15.32@15.44 a@15.44@15.54 half@15.54@15.8 "
     "percent.@15.8@16.26",
     [(None, [2000, 197000], ["2K", "197K"], [2.32, 2.98], "followers"),
      (5.6, [19.5], ["19.5%"], [5.44], "sales"),
      (8.5, [500000, 3e6], ["$500K", "$3M"], [9.7, 7.88], ""),
      (12.0, [10], ["10x"], [11.8], "faster"),
      (14.5, [12, 3.5], ["12%", "3.5%"], [14.06, 15.04], "churn")]),
]
TYPED = [
    ("2000,15000,197000", [2000, 15000, 197000], ["", "", ""]),
    ("2,000, 15,000, 197,000", [2000, 15000, 197000], ["", "", ""]),
    ("2,000,15,000", [2000, 15000], ["", ""]),
    ("[2000, 15000, 197000]", [2000, 15000, 197000], ["", "", ""]),
    ("Jan:2K, Feb:15K, Mar:197K", [2000, 15000, 197000], ["Jan", "Feb", "Mar"]),
    ("$2K | $197K", [2000, 197000], ["", ""]),
    ("12%; 3.5%", [12, 3.5], ["", ""]),
    ("1.5M 2.25M", [1.5e6, 2.25e6], ["", ""]),
]
UNITS = [
    # (typed values, the value, how it is written as it lands, and while counting toward the last)
    ("2K,197K", 197000, "197K", "197K"),
    ("2K,197K", 57312, "57,312", "57K"),
    ("500,197000", 500, "500", "500"),
    ("2K,197.5K", 197500, "197.5K", "197.5K"),
    ("1.25M", 1250000, "1.25M", "1.25M"),
    ("12%,3.5%", 7.31, "7.31%", "7.3%"),
    ("$2,000, $15,000", 15000, "$15K", "$15K"),
    ("950,1200", 1200, "1,200", "1,200"),
]


def main() -> int:
    failed = 0

    def check(ok: bool, what: str) -> None:
        nonlocal failed
        print(("ok    " if ok else "FAIL  ") + what)
        failed += 0 if ok else 1

    for line, values, texts, label, pattern in CASES:
        ws = words_of(line)
        said = am.spoken(ws)
        run = am.series(said, ws)
        got = [a.value for a in run.amounts] if run else None
        check(got is not None and [round(v, 6) for v in got] == [round(v, 6) for v in values],
              f"{line!r}: values {got} (want {values})")
        if not run:
            continue
        unit = am.Unit.of(run.amounts)
        written = [am.text(unit, a.value) for a in run.amounts]
        check(written == texts, f"    written {written} (want {texts})")
        lab = am.label_for(run, ws)
        check(lab == label, f"    label {lab!r} (want {label!r})")
        check(run.pattern == pattern, f"    pattern {run.pattern!r} (want {pattern!r})")
        times_ok = all(abs(a.start - ws[a.first].start) < 1e-9 for a in run.amounts)
        check(times_ok, "    each amount keeps the time of the word that says it")
    for transcript, runs in REAL:
        ws = [Word(t, float(s), float(e)) for t, s, e in (item.rsplit("@", 2) for item in transcript.split())]
        said = am.spoken(ws)
        for near, values, texts, at, label in runs:
            run = am.series(said, ws, near=near)
            got = [a.value for a in run.amounts] if run else None
            where = f"real transcript, near {near} s" if near is not None else "real transcript"
            check(got is not None and [round(v, 6) for v in got] == [round(v, 6) for v in values],
                  f"{where}: values {got} (want {values})")
            if not run:
                continue
            unit = am.Unit.of(run.amounts)
            written = [am.text(unit, a.value) for a in run.amounts]
            check(written == texts, f"    written {written} (want {texts})")
            starts = [round(a.start, 2) for a in run.amounts]
            check(starts == at, f"    lands at {starts} s (want {at}): the first piece of each figure")
            lab = am.label_for(run, ws)
            check(lab == label, f"    label {lab!r} (want {label!r})")
    for line in NOTHING:
        ws = words_of(line)
        run = am.series(am.spoken(ws), ws)
        check(run is None, f"{line!r}: nothing to chart (got {[a.said for a in run.amounts] if run else None})")
    for text, values, labels in TYPED:
        got = am.parse_list(text)
        check([round(a.value, 6) for a in got] == [round(v, 6) for v in values] and [a.label for a in got] == labels,
              f"typed {text!r}: {[a.value for a in got]} {[a.label for a in got]}")
    for text, v, landed, counting in UNITS:
        amounts = am.parse_list(text)
        unit = am.Unit.of(amounts)
        a = am.Unit.join(unit.exact(v))
        b = am.Unit.join(unit.during(v, amounts[-1].value))
        check(a == landed and b == counting, f"unit of {text!r}: {v} lands as {a!r} (want {landed!r}), "
                                             f"counts as {b!r} (want {counting!r})")
    for bad in ("two thousand", "12 apples", "", "$"):
        try:
            am.parse(bad)
            check(False, f"typed {bad!r} is refused")
        except ValueError as e:
            check(True, f"typed {bad!r} is refused: {e}")
    print(f"\n{'all passed' if not failed else f'{failed} failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
