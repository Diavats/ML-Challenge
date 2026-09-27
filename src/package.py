"""Build the final submission zip in the exact structure the challenge asks for:

  Coding_Bots_submission.zip
  ├── output/
  │   ├── matching_results.tsv        (the chosen version, e.g. v3)
  │   └── candidate_pairs.tsv
  ├── code/business_entity_resolution/
  │   ├── src/  README.md  requirements.txt  METHODOLOGY.md
  └── Documentation_template.md       (filled in)

Run:  python -m src.package v3
"""
import sys
import zipfile

from src.load import OUT, ROOT

TEAM = "Coding_Bots"


def main(version):
    matching = ROOT / "submissions" / f"{version}_matching_results.tsv"
    zip_path = ROOT / f"{TEAM}_submission.zip"
    code = "code/business_entity_resolution/"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(matching, "output/matching_results.tsv")
        z.write(OUT / "candidate_pairs.tsv", "output/candidate_pairs.tsv")
        for f in sorted((ROOT / "src").glob("*.py")):          # all source code
            z.write(f, code + "src/" + f.name)
        for name in ("README.md", "requirements.txt", "METHODOLOGY.md"):
            z.write(ROOT / name, code + name)
        z.write(ROOT / "Documentation_template.md", "Documentation_template.md")

    # check: list what is inside and make sure nothing is missing
    names = zipfile.ZipFile(zip_path).namelist()
    need = ["output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md",
            code + "README.md", code + "requirements.txt", code + "src/model.py"]
    missing = [n for n in need if n not in names]
    assert not missing, f"missing from zip: {missing}"
    print(f"{zip_path.name}: {len(names)} files, {zip_path.stat().st_size / 1e6:.0f} MB, uses {version}")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main(sys.argv[1])
