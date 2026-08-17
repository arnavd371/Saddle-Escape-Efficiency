"""
Bundles see_core.py + optimizers.py + a kernel-specific driver.py into a
single self-contained script (Kaggle script kernels don't support local
multi-file imports), and writes kernel-metadata.json for `kaggle kernels push`.

Usage: python build_kernel.py <kernel_dir_name>
  e.g. python build_kernel.py ext1_optimizers
Expects opt2026_ext/kernels/<name>/driver.py to exist.
Writes opt2026_ext/kernels/<name>/<name>.py (bundled) and kernel-metadata.json.
"""
import sys, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
KAGGLE_USERNAME = 'arnavd371'


def build(kernel_name, title=None, enable_gpu=True, enable_internet=False, machine_shape='NvidiaTeslaT4',
          extra_libs=()):
    kdir = os.path.join(HERE, 'kernels', kernel_name)
    driver_path = os.path.join(kdir, 'driver.py')
    if not os.path.exists(driver_path):
        raise FileNotFoundError(driver_path)

    libs = ['see_core.py', 'optimizers.py'] + list(extra_libs)
    parts = []
    for lib in libs:
        with open(os.path.join(HERE, lib)) as f:
            parts.append(f"# ==== bundled from {lib} ====\n" + f.read())
    with open(driver_path) as f:
        parts.append(f"# ==== driver: {kernel_name} ====\n" + f.read())

    bundled = "\n\n".join(parts)
    out_py = os.path.join(kdir, f'{kernel_name}.py')
    with open(out_py, 'w') as f:
        f.write(bundled)

    meta = {
        "id": f"{KAGGLE_USERNAME}/{kernel_name.replace('_', '-')}",
        "title": title or kernel_name.replace('_', ' '),
        "code_file": f"{kernel_name}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_internet": enable_internet,
        "machine_shape": machine_shape if enable_gpu else "",
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": []
    }
    with open(os.path.join(kdir, 'kernel-metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Bundled -> {out_py}  ({len(bundled)} chars)")
    print(f"Metadata -> {os.path.join(kdir, 'kernel-metadata.json')}")
    print(f"Kernel id -> {meta['id']}")
    return kdir


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: python build_kernel.py <kernel_dir_name> [title]")
        sys.exit(1)
    build(sys.argv[1], title=sys.argv[2] if len(sys.argv) > 2 else None)
