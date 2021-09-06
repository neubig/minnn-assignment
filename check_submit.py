#

# to check that the submiting zip file if well organized

import sys
import zipfile

def main(file: str):
    target_prefix = None
    required_files = {'minnn.py', 'classifier.py', 'sst-dev-output.txt', 'sst-test-output.txt', 'cfimdb-dev-output.txt', 'cfimdb-test-output.txt'}
    # --
    inside_files = set()
    with zipfile.ZipFile(file, 'r') as zz:
        # --
        print(f"Read zipfile {file}:")
        zz.printdir()
        print("#--")
        # --
        for info in zz.infolist():
            if info.filename.startswith("_"):
                continue  # ignore these files
            if target_prefix is None:
                target_prefix, _ = info.filename.split("/", 1)
                target_prefix = target_prefix + "/"
            # --
            assert info.filename.startswith(target_prefix), \
                'There should only be one top-level dir (with your andrew id as the dir-name) inside the zip file.'
            ff = info.filename[len(target_prefix):]
            inside_files.add(ff)
        # --
    # --
    required_files.difference_update(inside_files)
    assert len(required_files)==0, f"Required file not found: {required_files}"
    # --
    print(f"Read zipfile {file}, please check that your andrew-id is: {target_prefix[:-1]}")
    print(f"And it contains the following files: {sorted(list(inside_files))}")
    # --

if __name__ == '__main__':
    main(*sys.argv[1:])
