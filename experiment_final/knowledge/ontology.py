"""ICD-10-CM hierarchy loader + ancestors() (PIPELINE P2).

Reuses archive/experiments_caren/knowledge/ontology.py unchanged: it is loaded by file path (importlib)
because both experiment trees name their package `knowledge`, so a plain import would collide.
The tabular XML is read from the extracted CARE-N copy when present, otherwise extracted from the
ICD-10-CM 2027 Table-and-Index zip named in config.yaml into data/processed/.
"""
from __future__ import annotations

import importlib.util
import zipfile
from functools import lru_cache
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
REPO = EXP_DIR.parent
_CAREN_ONTO = REPO / "archive" / "experiments_caren" / "knowledge" / "ontology.py"
_EXTRACTED = REPO / "archive" / "experiments_caren" / "data" / "raw" / "icd10cm-table-and-index-2027" / "icd10cm-tabular_-2027.xml"


def _caren_module():
    """Import archive/experiments_caren/knowledge/ontology.py under a private name (P2, reuse)."""
    spec = importlib.util.spec_from_file_location("caren_ontology", _CAREN_ONTO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_M = _caren_module()
ROOT_NODE = _M.ROOT_NODE
Ontology = _M.Ontology


def tabular_xml(zip_path: str | None = None) -> Path:
    """Locate (or extract) the ICD-10-CM tabular XML (P2)."""
    if _EXTRACTED.exists():
        return _EXTRACTED
    out = EXP_DIR / "data" / "processed" / "icd10cm-tabular-2027.xml"
    if not out.exists():
        with zipfile.ZipFile(REPO / "experiment_final" / zip_path) as z:
            name = next(n for n in z.namelist() if n.endswith(".xml") and "tabular" in n)
            out.write_bytes(z.read(name))
    return out


@lru_cache(maxsize=1)
def get_ontology(zip_path: str | None = None) -> "Ontology":
    """Parsed ICD-10-CM 2027 hierarchy (chapters -> blocks -> categories -> codes) (P2)."""
    return _M.load_ontology(str(tabular_xml(zip_path)))


def resolve_code(onto: "Ontology", code: str):
    """Most specific existing node for an NHANES reason code; strips the NCHS 'P' suffix (P2)."""
    c = str(code).strip().upper()
    if not c or c.startswith("99999"):
        return None
    c = c.rstrip("P").rstrip(".")
    return onto.resolve(c)
