from __future__ import annotations
import argparse, io, zipfile
import numpy as np

def main():
    ap = argparse.ArgumentParser(description="Inspect WA-RSBE NPZ dataset archives")
    ap.add_argument("zip", help="ZIP archive containing one or more .npz files")
    args = ap.parse_args()
    with zipfile.ZipFile(args.zip) as zf:
        for name in zf.namelist():
            if not name.endswith('.npz'):
                continue
            print(f"\n[{name}]")
            with np.load(io.BytesIO(zf.read(name)), allow_pickle=True) as d:
                for k in d.files:
                    v = d[k]
                    print(f"{k:32s} shape={getattr(v,'shape',None)!s:18s} dtype={getattr(v,'dtype',None)}")
                if 'X26_columns' in d:
                    print('X26_columns:', d['X26_columns'].tolist())
                if 'Y_columns' in d:
                    print('Y_columns:', d['Y_columns'].tolist())
                if 'source_case_ids' in d:
                    print('cases:', d['source_case_ids'].tolist())

if __name__ == '__main__':
    main()
