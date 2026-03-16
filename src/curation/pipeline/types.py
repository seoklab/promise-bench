from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

class _PipelineModel(BaseModel):
    """Permissive base for all pipeline models."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

class AssemblyRow(_PipelineModel):
    """One row from ``_pdbx_struct_assembly`` in an mmCIF file."""

    id: str
    details: str = ""
    method_details: str = ""
    oligomeric_details: str = ""
    oligomeric_count: str = ""

    @property
    def is_author_defined(self) -> bool:
        return "author" in self.details.lower()

class CifMeta(_PipelineModel):
    """Metadata extracted from an mmCIF block header."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resolution: float
    method: str = ""
    keywords: str = ""
    block: object = Field(exclude=True)  # gemmi.cif.Block (opaque)

    def assembly_rows(self) -> list[AssemblyRow]:
        """Parse ``_pdbx_struct_assembly`` from the stored CIF block."""
        tbl = self.block.find_mmcif_category("_pdbx_struct_assembly")
        if not tbl:
            return []
        short_keys = [str(t).split(".")[-1] for t in tbl.tags]
        rows: list[AssemblyRow] = []
        for row in tbl:
            rec = {k: row[k] for k in short_keys}
            rows.append(AssemblyRow.model_validate(rec))
        return rows

    def label_base_to_entity_poly_type(self) -> dict[str, str]:
        """Map label_base → entity_poly.type from the CIF block."""
        from ..utils._pdb_helpers import label_base

        block = self.block
        label_ids = block.find_values("_struct_asym.id")
        ent_ids = block.find_values("_struct_asym.entity_id")
        lab2ent: dict[str, str] = {}
        if label_ids and ent_ids and len(label_ids) == len(ent_ids):
            for la, eid in zip(label_ids, ent_ids):
                if la and eid and la not in ("?", ".") and eid not in ("?", "."):
                    lab2ent[la] = eid

        ep_entity_id = block.find_values("_entity_poly.entity_id")
        ep_type = block.find_values("_entity_poly.type")
        ent2type: dict[str, str] = {}
        if ep_entity_id and ep_type and len(ep_entity_id) == len(ep_type):
            for eid, ty in zip(ep_entity_id, ep_type):
                if eid and ty:
                    ent2type[eid] = ty

        labbase2type: dict[str, str] = {}
        for la, eid in lab2ent.items():
            ty = ent2type.get(eid, "")
            labbase2type[label_base(la)] = ty
        return labbase2type

class AssemblyEntry(_PipelineModel):
    """One row of ``*_asm_raw.csv`` / ``*_asm_bio.csv``.

    Semicolon-separated list fields are stored as plain strings for CSV
    round-tripping; use the ``*_ids()`` helpers for typed access.
    """

    pdb: str
    chain_author: str
    chain_auth_asm: str
    assembly_id: str
    conf_label: str
    protein_count: int
    ligand_count: int
    chain_list_label: str  # semicolon-joined
    chain_list_author: str  # semicolon-joined
    ligands: str = ""  # semicolon-joined
    resolution: float
    experimental_method: str = ""
    desc: str = ""
    contact_chains: str = ""  # semicolon-joined
    contact_ligands: str = ""  # semicolon-joined

    # -- helpers for semicolon-list fields --

    def label_ids(self) -> list[str]:
        return [s for s in self.chain_list_label.split(";") if s]

    def author_ids(self) -> list[str]:
        return [s for s in self.chain_list_author.split(";") if s]

    def ligand_ids(self) -> list[str]:
        return [s for s in self.ligands.split(";") if s]

    def contact_chain_ids(self) -> list[str]:
        return [s for s in self.contact_chains.split(";") if s]

    def contact_ligand_ids(self) -> list[str]:
        return [s for s in self.contact_ligands.split(";") if s]

    @staticmethod
    def csv_headers() -> list[str]:
        return [
            "pdb",
            "chain_author",
            "chain_auth_asm",
            "assembly_id",
            "conf_label",
            "protein_count",
            "ligand_count",
            "chain_list_label",
            "chain_list_author",
            "ligands",
            "resolution",
            "experimental_method",
            "desc",
            "contact_chains",
            "contact_ligands",
        ]

    def to_csv_row(self) -> list:
        return [
            self.pdb,
            self.chain_author,
            self.chain_auth_asm,
            self.assembly_id,
            self.conf_label,
            self.protein_count,
            self.ligand_count,
            self.chain_list_label,
            self.chain_list_author,
            self.ligands,
            self.resolution,
            self.experimental_method,
            self.desc,
            self.contact_chains,
            self.contact_ligands,
        ]

    def is_homomer_duplicate(self, other: AssemblyEntry) -> bool:
        """True when *self* and *other* differ only by ``chain_auth_asm``."""
        return (
            self.assembly_id == other.assembly_id
            and self.conf_label == other.conf_label
            and self.chain_list_label == other.chain_list_label
            and self.chain_list_author == other.chain_list_author
            and self.ligands == other.ligands
            and self.chain_auth_asm != other.chain_auth_asm
        )

class AssemblyResult(_PipelineModel):
    """Result returned by ``fetch_assemblies``.

    Carries the gemmi structure and pre-extracted coordinates so that
    downstream code can compute contacts without re-parsing the CIF.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    assembly: object  # gemmi.Structure (opaque to pydantic)
    assembly_id: str
    bio_assembly: AssemblyRow  # parsed mmCIF assembly row
    chain_list_label: list[str]
    chain_list_author: list[str]
    ligand_list: list[str]
    polymer_count: int
    ligand_count: int
    resolution: float
    method: str
    desc: str
    assembly_auth_to_mmcif: dict[str, list[str]]
    chain_coords: dict[str, object] = Field(default_factory=dict)  # str → np.ndarray
    lig_instances: list[dict] = Field(default_factory=list)

    def to_entry(
        self,
        pdb: str,
        chain_author: str,
        chain_auth_asm: str,
        conf_label: str,
        *,
        contact_chains: list[str] | None = None,
        contact_ligands: list[str] | None = None,
    ) -> AssemblyEntry:
        """Build an :class:`AssemblyEntry`.

        Pass *contact_chains* / *contact_ligands* pre-computed by the
        caller; the model itself does **not** run geometry queries.
        """
        return AssemblyEntry(
            pdb=pdb,
            chain_author=chain_author,
            chain_auth_asm=chain_auth_asm,
            assembly_id=self.assembly_id,
            conf_label=conf_label,
            protein_count=self.polymer_count,
            ligand_count=self.ligand_count,
            chain_list_label=";".join(self.chain_list_label),
            chain_list_author=";".join(self.chain_list_author),
            ligands=";".join(self.ligand_list),
            resolution=self.resolution,
            experimental_method=self.method,
            desc=self.desc,
            contact_chains=";".join(contact_chains or []),
            contact_ligands=";".join(contact_ligands or []),
        )

class PairCall(_PipelineModel):
    """One row from ``pair-calls.csv``."""

    pdb: str
    assembly_id: str
    a_clone: str
    b_clone: str
    p_bio: float = 0.0
    label_raw: str = ""

    @field_validator("p_bio", mode="before")
    @classmethod
    def _coerce_p_bio(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0


class ProdigyEdge(_PipelineModel):
    """Classification result for one chain–chain edge."""

    pdb: str
    assembly_id: str
    chain_a: str
    chain_b: str
    p_bio: float
    final_label: str  # "BIO" | "LIG" | "XTAL"
    reason: str


class XtalEdge(_PipelineModel):
    pdb: str
    assembly_id: str
    chain_a: str
    chain_b: str
    reason: str
    p_bio: str


class LigandEdge(_PipelineModel):
    pdb: str
    assembly_id: str
    chain_a: str
    chain_b: str
    mediators: str  # semicolon-joined

class PairSide(_PipelineModel):
    """One side (a or b) of a conformational pair entry in the dataset CSV."""

    pdb: str
    assembly_id: str
    chain: str  # chain_auth_asm, e.g. "A1"
    conf_label: int
    chains: str = ""  # chain_list_author (semicolon-joined)
    contact_chains: str = ""  # semicolon-joined
    ligand_list: str = ""  # semicolon-joined
    contact_ligands: str = ""  # semicolon-joined
    desc: str = ""

    @field_validator("conf_label", mode="before")
    @classmethod
    def _coerce_conf_label(cls, v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0

    @property
    def chain_letter(self) -> str:
        """Extract chain letter from chain_auth_asm, e.g. ``'A1'`` → ``'A'``."""
        m = re.match(r"([A-Za-z]+)", self.chain)
        return m.group(1) if m else self.chain

    @property
    def is_monomer(self) -> bool:
        """True when entry has no protein contact partners."""
        return not bool(self.contact_chains)

    def chain_ids(self) -> list[str]:
        """Semicolon-split *chains* field."""
        return [s for s in self.chains.split(";") if s]

    def contact_chain_ids(self) -> list[str]:
        return [s for s in self.contact_chains.split(";") if s]

    def ligand_ids(self) -> list[str]:
        return [s for s in self.ligand_list.split(";") if s]

    def contact_ligand_ids(self) -> list[str]:
        return [s for s in self.contact_ligands.split(";") if s]

    def entity_key(self, suffix: str = "") -> str:
        """Build a unique key, e.g. ``'2wrz_1_A1'`` or ``'2wrz_1_A1_m'``."""
        key = f"{self.pdb.lower()}_{self.assembly_id}_{self.chain}"
        return f"{key}_{suffix}" if suffix else key


class DatasetPair(_PipelineModel):
    cluster_csv: str
    a: PairSide
    b: PairSide
    tm_score: Optional[float] = None

    @field_validator("tm_score", mode="before")
    @classmethod
    def _coerce_tm(cls, v):
        if v is None or (isinstance(v, str) and v.strip() in ("", "nan", "NaN")):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @property
    def cluster_stem(self) -> str:
        """``'A2/8A27_1'`` → ``'8A27_1'``."""
        return (
            self.cluster_csv.split("/")[-1]
            if "/" in self.cluster_csv
            else self.cluster_csv
        )

    @property
    def cluster_prefix(self) -> str:
        """``'A2/8A27_1'`` → ``'A2'``."""
        return (
            self.cluster_csv.split("/")[0]
            if "/" in self.cluster_csv
            else ""
        )

    @classmethod
    def from_csv_row(cls, row: dict) -> DatasetPair:
        def _s(v) -> str:
            """Normalise NaN / None to empty string."""
            if v is None:
                return ""
            s = str(v).strip()
            return "" if s in ("nan", "NaN", "None", "NONE") else s

        a = PairSide(
            pdb=_s(row.get("a_pdb")),
            assembly_id=_s(row.get("a_assembly_id")),
            chain=_s(row.get("a_chain")),
            conf_label=row.get("a_conf_label", 0),
            chains=_s(row.get("a_chains")),
            contact_chains=_s(row.get("a_contact_chains")),
            ligand_list=_s(row.get("a_ligand_list")),
            contact_ligands=_s(row.get("a_contact_ligands")),
            desc=_s(row.get("a_desc")),
        )
        b = PairSide(
            pdb=_s(row.get("b_pdb")),
            assembly_id=_s(row.get("b_assembly_id")),
            chain=_s(row.get("b_chain")),
            conf_label=row.get("b_conf_label", 0),
            chains=_s(row.get("b_chains")),
            contact_chains=_s(row.get("b_contact_chains")),
            ligand_list=_s(row.get("b_ligand_list")),
            contact_ligands=_s(row.get("b_contact_ligands")),
            desc=_s(row.get("b_desc")),
        )
        return cls(
            cluster_csv=_s(row.get("cluster_csv")),
            a=a,
            b=b,
            tm_score=row.get("tm_score"),
        )


    def to_csv_dict(self) -> dict[str, object]:
        """Flatten back to the CSV column layout produced by ``curate_sets``."""
        d: dict[str, object] = {"cluster_csv": self.cluster_csv}
        for prefix, side in (("a", self.a), ("b", self.b)):
            d[f"{prefix}_pdb"] = side.pdb
            d[f"{prefix}_assembly_id"] = side.assembly_id
            d[f"{prefix}_chain"] = side.chain
            d[f"{prefix}_conf_label"] = side.conf_label
            d[f"{prefix}_chains"] = side.chains
            d[f"{prefix}_contact_chains"] = side.contact_chains
            d[f"{prefix}_ligand_list"] = side.ligand_list
            d[f"{prefix}_contact_ligands"] = side.contact_ligands
        d["a_desc"] = self.a.desc
        d["b_desc"] = self.b.desc
        d["tm_score"] = self.tm_score
        return d

    CSV_COLUMNS: ClassVar[list[str]] = [
        "cluster_csv",
        "a_pdb", "a_assembly_id", "a_chain", "a_conf_label",
        "a_chains", "a_contact_chains", "a_ligand_list", "a_contact_ligands",
        "b_pdb", "b_assembly_id", "b_chain", "b_conf_label",
        "b_chains", "b_contact_chains", "b_ligand_list", "b_contact_ligands",
        "a_desc", "b_desc", "tm_score",
    ]

    @classmethod
    def to_dataframe(cls, pairs: list[DatasetPair]) -> pd.DataFrame:
        """Convert a list of pairs to a DataFrame with canonical column order."""
        if not pairs:
            return pd.DataFrame(columns=cls.CSV_COLUMNS)
        return pd.DataFrame([p.to_csv_dict() for p in pairs])[cls.CSV_COLUMNS]

    @classmethod
    def save_csv(cls, pairs: list[DatasetPair], path: str | Path) -> None:
        """Write pairs to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cls.to_dataframe(pairs).to_csv(path, index=False)

    @classmethod
    def load_csv(cls, path: str | Path) -> list[DatasetPair]:
        """Load all pairs from a dataset CSV file."""
        df = pd.read_csv(path)
        return [cls.from_csv_row(row) for _, row in df.iterrows()]

    @classmethod
    def load_csv_by_cluster(cls, path: str | Path) -> dict[str, list[DatasetPair]]:
        """Load pairs grouped by ``cluster_stem``."""
        pairs = cls.load_csv(path)
        groups: dict[str, list[DatasetPair]] = {}
        for p in pairs:
            groups.setdefault(p.cluster_stem, []).append(p)
        return groups

