# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()


root_path = Path(__file__).parent.parent.parent

sub_paths = [
    "data/src",
    "pipeline/src",
    "training/src",
    "utils/src",
    "verification/src",
    "plotting/src",
    "model/src",
]

for sub_path in sub_paths:
    sub_path = Path(root_path) / sub_path

    for path in sorted(Path(sub_path).rglob("*.py")):
        module_path = path.relative_to(sub_path).with_suffix("")
        doc_path = path.relative_to(sub_path).with_suffix(".md")
        full_doc_path = Path("reference", doc_path)

        parts = tuple(module_path.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue

        try:
            nav[parts] = doc_path.as_posix()
        except ValueError:
            continue

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root_path))

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
