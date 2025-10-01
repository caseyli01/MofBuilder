from pathlib import Path

def fetch_pdbfile(dir_name, keywords, nokeywords, ostream):
    candidates = []
    for pdb in Path(dir_name).rglob("*.pdb"):
        name = pdb.name
        if all(i in name for i in keywords) and all(
                j not in name for j in nokeywords):
            candidates.append(pdb.name)
    if len(candidates) == 0:
        raise ValueError(f"Cannot find a file including '{keywords}' ")
    elif len(candidates) == 1:
        ostream.print_info(f"Found the file including {keywords}: {candidates[0]}")
        return candidates
    elif len(candidates) > 1:
        ostream.print_info(f"Found many files including {keywords}: {candidates}")
        return candidates

