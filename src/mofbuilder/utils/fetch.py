from pathlib import Path

def fetch_pdbfile(dir_name, keywords, nokeywords):
    candidates = []
    for pdb in Path(dir_name).rglob("*.pdb"):
        name = pdb.name
        if all(i in name for i in keywords) and all(
                j not in name for j in nokeywords):
            candidates.append(pdb.name)
    if len(candidates) == 0:
        raise ValueError(f"Cannot find a file including '{keywords}' ")
    elif len(candidates) == 1:
        print("found the file including", keywords)
        return candidates
    elif len(candidates) > 1:
        print("found many files including", keywords)
        return candidates

def copy_file(old_path, new_path):
    src = Path(old_path)
    dest = Path(new_path)

    if (not dest.is_file()) or (not src.samefile(dest)):
        if not dest.parent.is_dir():
            dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text())

    print(f"File copied from {old_path} to {new_path}")

