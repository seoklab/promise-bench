import logging
import subprocess as sp
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional

import click
import numpy as np
import nuri
from joblib import Parallel, delayed
from nuri import fmt
from nuri.core import SubstructureCategory as Cat
from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    ValidationError,
    field_validator,
)
from tqdm import tqdm

from ..utils._data_root import DataRootCommand
from ..utils.typedefs import GroupSet


class ResidueId(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    pdb_seq_num: int
    pdb_ins_code: str = ""

    @classmethod
    def from_mmcif(cls, data: dict[str, Optional[str]]):
        try:
            return cls.model_validate(data)
        except ValidationError:
            return None

    @field_validator("pdb_ins_code", mode="before")
    @staticmethod
    def _coerce_ins_code(v):
        return v or ""


@dataclass
class ClusterMember:
    pdb_id: str
    entity_id: str
    chain_id: str

    seq: str
    res_ids: list[Optional[ResidueId]]

    def as_fasta(self):
        return f">{self.pdb_id}_{self.chain_id}\n{self.seq}\n"

    @classmethod
    def from_member_spec(
        cls, spec: str, onerror: list[str], mmcif_dir: Path = Path(".")
    ) -> list["ClusterMember"]:
        pdb_id, entity_idx_str = spec.split("_")
        pdb_id = pdb_id.lower()

        try:
            frame = next(
                fmt.read_cif(str(mmcif_dir / f"{pdb_id}.cif"))
            ).data
        except FileNotFoundError:
            onerror.append(spec)
            return []

        mmcif = fmt.cif_ddl2_frame_as_dict(frame)

        # The FASTA was built from pdb_seqres.txt (protein-only),
        # numbering unique protein sequences per PDB sequentially.
        # Protein-NA complexes are filtered out upstream, so for
        # remaining protein-only PDBs the sequential index equals
        # the CIF entity_id.
        entity_id = entity_idx_str

        try:
            chain_ids = entity_find_chain_all(mmcif, entity_id)
        except ValueError:
            onerror.append(spec)
            return []

        return [
            cls(pdb_id, entity_id, cid, *mmcif_seqres(mmcif, cid))
            for cid in chain_ids
        ]


MsaResSeq = dict[ResidueId, int]


def _three_to_one(three: str):
    mapping = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    return mapping.get(three.upper(), "X")


def entity_find_chain_all(
    cif_dict: dict[str, list[dict[str, Optional[str]]]],
    entity_id: str,
):
    chains = [
        ch
        for entity in cif_dict["entity_poly"]
        if entity["entity_id"] == entity_id
        and (chains := entity.get("pdbx_strand_id"))
        for ch in chains.split(",")
    ]
    if not chains:
        raise ValueError(f"Entity id {entity_id} not found in the CIF file.")

    return chains


def mmcif_seqres(
    mmcif: dict[str, list[dict[str, Optional[str]]]],
    chain_id: str,
):
    seq: list[str] = []
    res_ids: list[Optional[ResidueId]] = []

    for scheme in mmcif["pdbx_poly_seq_scheme"]:
        if scheme["pdb_strand_id"] != chain_id:
            continue

        seq.append(_three_to_one(scheme["mon_id"]))
        res_ids.append(ResidueId.from_mmcif(scheme))

    return "".join(seq), res_ids


def famsa_create_msa(seqs: Path, result: Path, nthreads: int):
    ret = sp.run(
        ["famsa", "-t", str(nthreads), seqs, result],
        check=False,
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
    )
    if ret.returncode != 0:
        raise RuntimeError(f"Error processing {seqs.stem}: {ret.stdout.decode()}")


def create_mapping(
    center: str,
    members: dict[tuple[str, str], ClusterMember],
    msa: Path,
):
    msa_resseq: dict[tuple[str, str], MsaResSeq] = {}

    with open(msa) as f:
        line = next(f)

        while line.startswith(">"):
            pdb_id, chain_id = line[1:].strip().split("_")
            member = members[(pdb_id, chain_id)]

            resseq = msa_resseq[(pdb_id, chain_id)] = {}

            fragments = []
            for line in f:
                if line.startswith(">"):
                    break
                fragments.append(line.strip())

            mit = iter(member.res_ids)
            for i, res in enumerate(chain(*fragments)):
                if res == "-":
                    continue

                rid = next(mit)
                if rid is not None:
                    resseq[rid] = i

            if next(mit, None) is not None:
                raise ValueError(
                    (
                        "Residue count mismatch between sequence "
                        f"and MSA: {center}, {pdb_id}_{chain_id}"
                    )
                )

    return msa_resseq


def load_cif_chains(
    pdb_id: str,
    chain_mappings: Iterable[tuple[str, MsaResSeq]],
    mmcif_dir: Path = Path("."),
):
    path = str(mmcif_dir / f"{pdb_id}.cif")

    mol = next(nuri.readfile("cif", path, sanitize=False))

    atom_residue_map: dict[int, ResidueId] = {
        atom.as_parent().id: ResidueId(
            pdb_seq_num=sub.id, pdb_ins_code=sub.props.get("icode", "")
        )
        for sub in mol.subs
        if sub.category == Cat.Residue
        for atom in sub
    }

    results: list[tuple[str, np.ndarray, dict[int, int]]] = []
    for cid, resseq in chain_mappings:
        sub = next(
            sub
            for sub in mol.subs
            if sub.category == Cat.Chain and sub.name == cid
        )

        atom_is_ca = [
            atom.atomic_number == 6 and atom.name == "CA" for atom in sub
        ]
        coords = sub.get_conf()[atom_is_ca]

        ca_residues = [
            atom_residue_map[atom.as_parent().id]
            for atom, is_ca in zip(sub, atom_is_ca)
            if is_ca
        ]
        align_to_idx = {
            mi: ai
            for ai, r in enumerate(ca_residues)
            # non-polymer entity can share chain
            if (mi := resseq.get(r, None)) is not None
        }

        if not align_to_idx:
            print(
                f"Warning: No residues aligned for {pdb_id}_{cid}",
                flush=True,
            )
            continue

        results.append((f"{pdb_id}_{cid}", coords, align_to_idx))

    return results


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def msa_align_coords(
    cluster: list[str],
    seqs: Path,
    msa: Path,
    result: Path,
    missing: Path,
    nthreads: int,
    mmcif_dir: Path = Path("."),
):
    logging.basicConfig(level=logging.FATAL)

    if _nonempty(seqs) and _nonempty(msa) and _nonempty(result):
        return

    seqs.parent.mkdir(exist_ok=True)
    msa.parent.mkdir(exist_ok=True)
    result.parent.mkdir(exist_ok=True)

    onerror: list[str] = []

    members: dict[tuple[str, str], ClusterMember] = {
        (member.pdb_id, member.chain_id): member
        for spec in cluster
        for member in ClusterMember.from_member_spec(spec, onerror, mmcif_dir)
    }

    if onerror:
        missing.parent.mkdir(exist_ok=True)
        missing.write_text("\n".join(m for m in onerror) + "\n")

    if not members:
        return

    with open(seqs, "w") as f:
        for member in members.values():
            f.write(member.as_fasta())

    famsa_create_msa(seqs, msa, nthreads)

    resseqs = create_mapping(cluster[0], members, msa)

    grouped_resseqs: dict[str, dict[str, MsaResSeq]] = defaultdict(dict)
    for (pdb_id, chain_id), resseq in resseqs.items():
        grouped_resseqs[pdb_id][chain_id] = resseq

    mapped_coords = np.array(
        [
            entry
            for pdb_id, pdb_resseqs in grouped_resseqs.items()
            for entry in load_cif_chains(pdb_id, pdb_resseqs.items(), mmcif_dir)
        ],
        dtype=object,
    )

    np.savez_compressed(result, coords=mapped_coords)


@click.command(cls=DataRootCommand)
@click.option("--nproc", "-n", type=int, default=8)
@click.option(
    "--mmcif-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing PDB mmCIF files (external input).",
)
@click.option(
    "--seqs",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/seqs"),
    show_default=True,
)
@click.option(
    "--msas",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/msas"),
    show_default=True,
)
@click.option(
    "--coords",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/coords"),
    show_default=True,
)
@click.option(
    "--missing",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/missing"),
    show_default=True,
)
@click.argument(
    "spec",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def main(
    nproc: int,
    mmcif_dir: Path,
    seqs: Path,
    msas: Path,
    coords: Path,
    missing: Path,
    spec: Path,
):
    clusters = TypeAdapter(list[GroupSet]).validate_json(spec.read_bytes())

    seqs.mkdir(exist_ok=True, parents=True)
    msas.mkdir(exist_ok=True, parents=True)
    coords.mkdir(exist_ok=True, parents=True)
    missing.mkdir(exist_ok=True, parents=True)

    Parallel(n_jobs=nproc)(
        delayed(msa_align_coords)(
            [
                cluster.center,
                *(m for m in cluster.members if m != cluster.center),
            ],
            seqs / cluster.center[1:3] / f"{cluster.center}.fa",
            msas / cluster.center[1:3] / f"{cluster.center}.a3m",
            coords / cluster.center[1:3] / f"{cluster.center}.npz",
            missing / cluster.center[1:3] / f"{cluster.center}.txt",
            1,
            mmcif_dir,
        )
        for cluster in tqdm(clusters)
    )


if __name__ == "__main__":
    main()
