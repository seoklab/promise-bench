#!/usr/bin/env python3
"""
CIF to Renumbered PDB Converter for ProMiSE-bench.

Convert CIF files to renumbered PDB based on MSA alignment.
Residue numbers are assigned based on representative sequence position.

Usage:
    # Process all CIF files in examples/targets
    python -m src.eval.cif_to_renumbered_pdb --targets-dir examples/targets

    # Process specific CIF file
    python -m src.eval.cif_to_renumbered_pdb --input examples/targets/intrinsic/7OYW_1/asm_6yeb_1.cif --cluster 7OYW_1 --set intrinsic
"""

import os
import re
import json
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
from Bio.PDB import MMCIFParser, PDBIO, Select
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from utils._config import pipeline_cfg as C
from utils._config import eval_cfg as E

AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'MSE': 'M', 'SEC': 'C', 'PYL': 'K'
}


# ============================================================================
# MSA Parsing
# ============================================================================

def parse_a3m(a3m_path: str) -> Dict[str, str]:
    """Parse a3m file. Returns {header: aligned_sequence (no lowercase)}."""
    sequences = {}
    header, seq = None, []
    
    with open(a3m_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    sequences[header] = ''.join(seq)
                header, seq = line[1:], []
            elif line:
                # Remove lowercase (insertions), keep uppercase and gaps
                seq.append(re.sub(r'[a-z]', '', line))
        if header:
            sequences[header] = ''.join(seq)
    
    return sequences


def build_position_dict(aligned_seq: str) -> Dict[int, str]:
    """
    Build position dict from aligned sequence.
    Returns {alignment_column (1-based): residue} for non-gap positions.
    """
    pos_dict = {}
    for col, char in enumerate(aligned_seq, 1):
        if char != '-':
            pos_dict[col] = char
    return pos_dict


def create_renumber_mapping(
    rep_aligned: str,
    target_aligned: str
) -> Dict[int, Optional[int]]:
    """
    Create mapping from target residue position to rep-based residue number.
    
    Returns:
        {target_residue_pos (1-based): new_resnum or None (to remove)}
    
    Logic:
    - Iterate through alignment columns
    - For each column where target has residue:
      - If rep has residue: map to rep's residue position
      - If rep has gap: mark as None (remove)
    """
    if len(rep_aligned) != len(target_aligned):
        raise ValueError(f"Alignment length mismatch: {len(rep_aligned)} vs {len(target_aligned)}")
    
    mapping = {}
    rep_pos = 0  # rep residue position (1-based, counting non-gaps)
    target_pos = 0  # target residue position (1-based, counting non-gaps)
    
    for col in range(len(rep_aligned)):
        rep_char = rep_aligned[col]
        target_char = target_aligned[col]
        
        # Count rep position
        if rep_char != '-':
            rep_pos += 1
        
        # Count target position and create mapping
        if target_char != '-':
            target_pos += 1
            if rep_char != '-':
                # Both have residue - map target to rep position
                mapping[target_pos] = rep_pos
            else:
                # Rep has gap - remove this residue
                mapping[target_pos] = None
    
    return mapping


# ============================================================================
# CIF Processing
# ============================================================================

def get_auth_chain_residues(cif_path: str, auth_asym_id: str) -> Tuple[str, List[Tuple], str]:
    """
    Extract sequence and residue info from CIF for specific auth_asym_id.
    
    Note: BioPython MMCIFParser uses auth_asym_id as chain ID.
    
    Returns:
        (sequence, [(residue_obj, old_resnum), ...], chain_id_in_structure)
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('s', cif_path)
    
    # BioPython uses auth_asym_id as chain ID
    # Find available chains
    available_chains = {}
    for model in structure:
        for chain in model:
            res_count = sum(1 for r in chain if r.get_id()[0] == ' ')
            if res_count > 0:
                available_chains[chain.get_id()] = res_count
    
    # Try to find target chain
    target_chain_id = None
    
    # First try exact match
    if auth_asym_id in available_chains:
        target_chain_id = auth_asym_id
    else:
        # Try without number suffix (A1 -> A)
        auth_letter = auth_asym_id[0] if auth_asym_id else None
        if auth_letter in available_chains:
            target_chain_id = auth_letter
        else:
            # Use first available polymer chain
            if available_chains:
                target_chain_id = list(available_chains.keys())[0]
                print(f"  WARNING: auth_asym_id {auth_asym_id} not found")
                print(f"  Available chains: {list(available_chains.keys())}")
    
    if target_chain_id is None:
        print(f"  ERROR: No polymer chains found in structure")
        return '', [], ''
    
    print(f"  Using chain: {target_chain_id} (requested: {auth_asym_id})")
    
    # Extract residues
    sequence = []
    residue_info = []
    
    for model in structure:
        for chain in model:
            if chain.get_id() == target_chain_id:
                for residue in chain:
                    if residue.get_id()[0] == ' ':  # Standard residue
                        res_name = residue.get_resname()
                        one_letter = AA_3TO1.get(res_name, 'X')
                        sequence.append(one_letter)
                        residue_info.append((residue, residue.get_id()[1]))
                break
        break
    
    return ''.join(sequence), residue_info, target_chain_id


class ChainSelect(Select):
    """Select only specific chain."""
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id


def build_auth_to_label_seq_id(cif_path: str, auth_asym_id: str) -> Dict[int, int]:
    """
    Build mapping from auth_seq_id to label_seq_id for a given chain.
    
    BioPython uses auth_seq_id for residue numbering, but label_seq_id
    corresponds to the a3m MSA position (1-based, sequential).
    """
    mmcif_dict = MMCIF2Dict(cif_path)
    
    auth_seq_ids = mmcif_dict.get('_atom_site.auth_seq_id', [])
    label_seq_ids = mmcif_dict.get('_atom_site.label_seq_id', [])
    auth_asym_ids = mmcif_dict.get('_atom_site.auth_asym_id', [])
    atom_names = mmcif_dict.get('_atom_site.label_atom_id', [])
    
    if isinstance(auth_seq_ids, str):
        auth_seq_ids = [auth_seq_ids]
        label_seq_ids = [label_seq_ids]
        auth_asym_ids = [auth_asym_ids]
        atom_names = [atom_names]
    
    mapping = {}
    for i in range(len(auth_seq_ids)):
        if auth_asym_ids[i] == auth_asym_id and atom_names[i] == 'CA':
            try:
                auth_id = int(auth_seq_ids[i])
                label_id = int(label_seq_ids[i])
                mapping[auth_id] = label_id
            except (ValueError, TypeError):
                continue
    
    return mapping


def cif_to_renumbered_pdb(
    cif_path: str,
    output_pdb: str,
    auth_asym_id: str,
    renumber_mapping: Dict[int, Optional[int]],
    new_chain_id: str = 'A'
) -> bool:
    """
    Convert CIF to renumbered PDB.
    
    Uses label_seq_id (= a3m position) as key into renumber_mapping,
    NOT auth_seq_id (which BioPython uses by default).
    
    Args:
        cif_path: Input CIF file
        output_pdb: Output PDB file
        auth_asym_id: Auth chain ID to extract
        renumber_mapping: {target_pos (label_seq_id): new_resnum or None}
        new_chain_id: Chain ID for output PDB
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure('s', cif_path)
    
    # Get the sequence and residue info
    seq, residue_info, target_chain_id = get_auth_chain_residues(cif_path, auth_asym_id)
    
    if not residue_info:
        print(f"  ERROR: No residues found for {auth_asym_id}")
        return False
    
    # Build auth_seq_id -> label_seq_id mapping
    auth_to_label = build_auth_to_label_seq_id(cif_path, target_chain_id)
    
    # Create new structure with renumbered residues
    new_structure = copy.deepcopy(structure)
    
    renumbered = 0
    removed = 0
    
    for model in new_structure:
        for chain in model:
            if chain.get_id() != target_chain_id:
                continue
            
            residue_data = []
            
            for residue in list(chain.get_residues()):
                if residue.get_id()[0] != ' ':
                    continue
                
                old_id = residue.get_id()
                auth_resnum = old_id[1]  # BioPython gives auth_seq_id
                
                # Convert to label_seq_id (= a3m position)
                label_resnum = auth_to_label.get(auth_resnum, auth_resnum)
                
                if label_resnum in renumber_mapping:
                    new_resnum = renumber_mapping[label_resnum]
                    if new_resnum is not None:
                        new_id = (old_id[0], new_resnum, old_id[2])
                        residue_data.append((residue, old_id, new_id))
                    else:
                        residue_data.append((residue, old_id, None))  # Remove
                else:
                    residue_data.append((residue, old_id, None))  # Not in mapping
            
            # Detach all residues
            for residue, old_id, _ in residue_data:
                chain.detach_child(old_id)
            
            # Re-add with new IDs (sorted by new resnum)
            valid_residues = [(r, oid, nid) for r, oid, nid in residue_data if nid is not None]
            valid_residues.sort(key=lambda x: x[2][1])
            
            for residue, old_id, new_id in valid_residues:
                residue.id = new_id
                chain.add(residue)
                renumbered += 1
            
            removed = len(residue_data) - len(valid_residues)
            
            # Change chain ID for PDB format
            chain.id = new_chain_id
    
    print(f"  Renumbered: {renumbered}, Removed: {removed}")
    
    # Save PDB
    os.makedirs(os.path.dirname(output_pdb), exist_ok=True)
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb, ChainSelect(new_chain_id))
    
    return True


# ============================================================================
# Main Processing
# ============================================================================

def parse_answer_tag(tag: str) -> Tuple[str, str, str]:
    """
    Parse answer tag like '6yeb_1_A1_conf_0'.
    Returns (pdb_id, asm_num, auth_asym_id)
    """
    parts = tag.split('_')
    pdb_id = parts[0]
    asm_num = parts[1]
    auth_asym_id = parts[2]
    return pdb_id, asm_num, auth_asym_id


def get_msa_tag(pdb_id: str, auth_asym_id: str) -> str:
    """
    Convert pdb_id and auth_asym_id to MSA tag.
    e.g., ('6yeb', 'A1') -> '6yeb_A'
    """
    # Extract letter part from auth_asym_id (A1 -> A, B2 -> B)
    chain_letter = re.match(r'([A-Za-z]+)', auth_asym_id).group(1)
    return f"{pdb_id.lower()}_{chain_letter}"


def process_cluster(
    cluster_id: str,
    set_name: str,
    targets_dir: str,
    msa_dir: str,
    rep_seq_data: Dict,
    answer_map_data: Dict,
    output_dir: str
) -> Dict[str, bool]:
    """
    Process all CIF files for a cluster.
    
    Returns: {cif_filename: success}
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Cluster: {cluster_id} ({set_name})")
    print(f"{'='*60}")
    
    # Get answer map for this cluster
    cluster_data = answer_map_data.get(set_name, {}).get(cluster_id, {})
    if not cluster_data:
        print(f"  ERROR: Cluster not found in answer_map")
        return results
    
    # Get representative sequence info
    rep_info = rep_seq_data.get(cluster_id, {})
    if not rep_info:
        print(f"  ERROR: Cluster not found in rep_seq")
        return results
    
    rep_header = rep_info['header']  # e.g., '7oyw_A'
    print(f"  Rep header: {rep_header}")
    
    # Load MSA
    msa_path = Path(msa_dir) / f"{cluster_id}.a3m"
    if not msa_path.exists():
        print(f"  ERROR: MSA not found: {msa_path}")
        return results
    
    msa_sequences = parse_a3m(str(msa_path))
    print(f"  MSA sequences: {len(msa_sequences)}")
    
    # Get rep aligned sequence
    rep_aligned = msa_sequences.get(rep_header)
    if rep_aligned is None:
        print(f"  ERROR: Rep header '{rep_header}' not found in MSA")
        return results
    
    # Process apo and holo targets
    all_tags = cluster_data.get('apo', []) + cluster_data.get('holo', [])
    
    for tag in all_tags:
        pdb_id, asm_num, auth_asym_id = parse_answer_tag(tag)
        msa_tag = get_msa_tag(pdb_id, auth_asym_id)
        
        print(f"\n  Processing: {tag}")
        print(f"    PDB: {pdb_id}, ASM: {asm_num}, Auth: {auth_asym_id}")
        print(f"    MSA tag: {msa_tag}")
        
        # Find CIF file - search by pdb_id since asm_num might differ
        cluster_dir = Path(targets_dir) / set_name / cluster_id
        cif_files = list(cluster_dir.glob(f"asm_{pdb_id}_*.cif"))
        
        if not cif_files:
            print(f"    ERROR: No CIF found for pdb_id={pdb_id} in {cluster_dir}")
            results[f"asm_{pdb_id}_*.cif"] = False
            continue
        
        cif_path = cif_files[0]  # Use first match
        actual_asm_num = cif_path.stem.split('_')[2]  # Get actual asm_num from filename
        print(f"    Using CIF: {cif_path.name}")
        
        # Get target aligned sequence from MSA
        target_aligned = msa_sequences.get(msa_tag)
        if target_aligned is None:
            # Try with uppercase
            for header in msa_sequences:
                if header.lower() == msa_tag.lower():
                    target_aligned = msa_sequences[header]
                    break
        
        if target_aligned is None:
            print(f"    ERROR: MSA tag '{msa_tag}' not found in MSA")
            print(f"    Available: {list(msa_sequences.keys())[:5]}...")
            results[cif_pattern] = False
            continue
        
        # Create renumber mapping
        try:
            renumber_mapping = create_renumber_mapping(rep_aligned, target_aligned)
            mapped_count = sum(1 for v in renumber_mapping.values() if v is not None)
            print(f"    Mapping: {len(renumber_mapping)} positions, {mapped_count} mapped")
        except Exception as e:
            print(f"    ERROR creating mapping: {e}")
            results[cif_pattern] = False
            continue
        
        # Output path - use actual asm_num from CIF filename
        output_pdb = Path(output_dir) / set_name / cluster_id / f"{pdb_id}_{actual_asm_num}_{auth_asym_id}_renumbered.pdb"
        
        # Convert
        try:
            success = cif_to_renumbered_pdb(
                str(cif_path),
                str(output_pdb),
                auth_asym_id,
                renumber_mapping
            )
            results[cif_path.name] = success
            if success:
                print(f"    Saved: {output_pdb.name}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[cif_path.name] = False
    
    return results


def discover_clusters(targets_dir: str) -> List[Tuple[str, str]]:
    """
    Discover all clusters from targets directory.
    Returns: [(set_name, cluster_id), ...]
    """
    clusters = []
    targets_path = Path(targets_dir)
    
    for set_dir in targets_path.iterdir():
        if not set_dir.is_dir():
            continue
        set_name = set_dir.name
        
        for cluster_dir in set_dir.iterdir():
            if not cluster_dir.is_dir():
                continue
            cluster_id = cluster_dir.name
            clusters.append((set_name, cluster_id))
    
    return sorted(clusters)


# ============================================================================
# CLI
# ============================================================================

@click.command()
@click.option('--targets-dir', type=click.Path(exists=True), default=None,
              help='examples/targets directory')
@click.option('--input', '-i', 'input_cif', type=click.Path(exists=True), default=None,
              help='Input CIF file')
@click.option('--cluster', type=str, default=None,
              help='Cluster ID (required with --input)')
@click.option('--set', 'set_name', type=str, default=None,
              help='Set name (required with --input)')
@click.option('--output-dir', '-o', type=click.Path(),
              default=str(E.dir('renumbered_pdbs')),
              show_default=True, help='Output directory for renumbered PDBs')
@click.option('--msa-dir', type=click.Path(exists=True),
              default=str(C.dir('msas')),
              show_default=True, help='MSA directory')
@click.option('--rep-seq', type=click.Path(exists=True),
              default=str(C.file('rep_seq')),
              show_default=True, help='Path to rep_seq.json')
@click.option('--answer-map', type=click.Path(exists=True),
              default=str(C.file('answer_map')),
              show_default=True, help='Path to answer_map.json')
def main(targets_dir, input_cif, cluster, set_name, output_dir, msa_dir,
         rep_seq, answer_map):
    """CIF to Renumbered PDB Converter for ProMiSE-bench."""
    if not targets_dir and not input_cif:
        raise click.UsageError('One of --targets-dir or --input is required.')

    # Load data files
    print("Loading data files...")

    with open(rep_seq) as f:
        rep_seq_data = json.load(f)
    print(f"  rep_seq: {len(rep_seq_data)} clusters")

    with open(answer_map) as f:
        answer_map_data = json.load(f)
    total_clusters = sum(len(v) for v in answer_map_data.values())
    print(f"  answer_map: {total_clusters} clusters")

    # Discover or use specified clusters
    if targets_dir:
        clusters = discover_clusters(targets_dir)
        print(f"\nFound {len(clusters)} clusters in {targets_dir}")
    else:
        if not cluster or not set_name:
            raise click.UsageError('--cluster and --set are required with --input.')
        clusters = [(set_name, cluster)]

    # Process clusters
    success_total = 0
    fail_total = 0

    for sn, cid in clusters:
        tdir = targets_dir if targets_dir else str(Path(input_cif).parent.parent.parent)

        results = process_cluster(
            cluster_id=cid,
            set_name=sn,
            targets_dir=tdir,
            msa_dir=msa_dir,
            rep_seq_data=rep_seq_data,
            answer_map_data=answer_map_data,
            output_dir=output_dir,
        )

        success_total += sum(1 for v in results.values() if v)
        fail_total += sum(1 for v in results.values() if not v)

    # Summary
    print(f"\n{'#'*60}")
    print("SUMMARY")
    print(f"{'#'*60}")
    print(f"Success: {success_total}")
    print(f"Failed: {fail_total}")

    raise SystemExit(0 if fail_total == 0 else 1)


if __name__ == "__main__":
    main()
