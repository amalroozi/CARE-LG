"""Downloader for pooled NHANES 1999-2023 (NHANES_POOLED.md; PIPELINE P1).

Fetches every component .xpt (+ its .htm codebook) from wwwn.cdc.gov and the public-use linked
mortality files (LMF, 2019 release) from ftp.cdc.gov into data/raw/nhanes_pooled/<cycle>/, skipping
files already present. File names vary by cycle; for each component the candidate names are tried in
order and the first one that returns a genuine SAS transport file (XPT magic header) is kept.
Writes data/raw/nhanes_pooled/download_log.csv (cycle, component, file, status, bytes).

  .venv/bin/python -m experiment_final.adapters.nhanes_pooled_download
"""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
RAW = EXP_DIR / "data" / "raw" / "nhanes_pooled"
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}"
LMF = "https://ftp.cdc.gov/pub/health_statistics/nchs/datalinkage/linked_mortality/NHANES_{a}_{b}_MORT_2019_PUBLIC.dat"
XPT_MAGIC = b"HEADER RECORD*******LIBRARY HEADER RECORD"

# cycle -> (url year, suffix, mortality years or None)
CYCLES = {
    "1999-2000": ("1999", "", ("1999", "2000")), "2001-2002": ("2001", "_B", ("2001", "2002")),
    "2003-2004": ("2003", "_C", ("2003", "2004")), "2005-2006": ("2005", "_D", ("2005", "2006")),
    "2007-2008": ("2007", "_E", ("2007", "2008")), "2009-2010": ("2009", "_F", ("2009", "2010")),
    "2011-2012": ("2011", "_G", ("2011", "2012")), "2013-2014": ("2013", "_H", ("2013", "2014")),
    "2015-2016": ("2015", "_I", ("2015", "2016")), "2017-2020": ("2017", "P_", ("2017", "2018")),
    "2021-2023": ("2021", "_L", None),
}


def candidates(component: str, suf: str) -> list:
    """Candidate file stems per component and cycle (NHANES_POOLED.md table + CDC naming) (P1)."""
    if suf == "P_":
        name = lambda s: f"P_{s}"  # noqa: E731
    else:
        name = lambda s: f"{s}{suf}"  # noqa: E731
    old = {"": ("LAB13", "LAB13", "LAB10", "LAB13AM"), "_B": ("L13_B", "L13_B", "L10_B", "L13AM_B"),
           "_C": ("L13_C", "L13_C", "L10_C", "L13AM_C")}
    if component == "TCHOL":
        return [old[suf][0]] if suf in old else [name("TCHOL")]
    if component == "HDL":
        return [old[suf][1]] if suf in old else [name("HDL")]
    if component == "GHB":
        return [old[suf][2]] if suf in old else [name("GHB")]
    if component == "BPX":
        return [name("BPXO"), name("BPX")] if suf in ("P_", "_L") else [name("BPX")]
    if component == "KIQ":
        return [name("KIQ_U"), name("KIQ")]
    return [name(component)]


COMPONENTS = ["DEMO", "BMX", "BPX", "BPQ", "DIQ", "SMQ", "MCQ", "KIQ", "TCHOL", "HDL", "GHB", "RXQ_RX"]


def _fetch(url: str, dest: Path, check_xpt: bool) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "present"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; CARE-LG)"})
        data = urllib.request.urlopen(req, timeout=180).read()
    except Exception as e:  # noqa: BLE001
        return f"error:{type(e).__name__}"
    if check_xpt and not data.startswith(XPT_MAGIC):
        return "not_xpt"
    dest.write_bytes(data)
    return "downloaded"


def download_all() -> list:
    """Download every cycle/component (+ codebook) and the LMF files; returns the log rows (P1)."""
    rows = []
    for cyc, (year, suf, mort) in CYCLES.items():
        d = RAW / cyc
        d.mkdir(parents=True, exist_ok=True)
        for comp in COMPONENTS:
            got = None
            for stem in candidates(comp, suf):
                st = _fetch(BASE.format(year=year, name=f"{stem}.xpt"), d / f"{stem}.xpt", True)
                rows.append({"cycle": cyc, "component": comp, "file": f"{stem}.xpt", "status": st,
                             "bytes": (d / f"{stem}.xpt").stat().st_size if (d / f"{stem}.xpt").exists() else 0})
                if st in ("present", "downloaded"):
                    got = stem
                    hs = _fetch(BASE.format(year=year, name=f"{stem}.htm"), d / f"{stem}.htm", False)
                    rows.append({"cycle": cyc, "component": comp, "file": f"{stem}.htm", "status": hs,
                                 "bytes": (d / f"{stem}.htm").stat().st_size if (d / f"{stem}.htm").exists() else 0})
                    break
            if got is None:
                rows.append({"cycle": cyc, "component": comp, "file": "", "status": "missing", "bytes": 0})
        if mort:
            f = f"NHANES_{mort[0]}_{mort[1]}_MORT_2019_PUBLIC.dat"
            st = _fetch(LMF.format(a=mort[0], b=mort[1]), d / f, False)
            rows.append({"cycle": cyc, "component": "MORT", "file": f, "status": st,
                         "bytes": (d / f).stat().st_size if (d / f).exists() else 0})
        print(cyc, {r["component"]: r["status"] for r in rows if r["cycle"] == cyc and not r["file"].endswith(".htm")}, flush=True)
    with open(RAW / "download_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["cycle", "component", "file", "status", "bytes"])
        w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    download_all()
