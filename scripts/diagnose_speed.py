import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root", type=str)
    p.add_argument("--batches", type=int, default=20,
                   help="Number of batches to measure (default: 20)")
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def hr(): print("-" * 55)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    print("=" * 55)
    print("  TRAINING EFFICIENCY DIAGNOSIS")
    print("=" * 55)
    print()

    print("[ 1 ] ENVIRONMENT")
    hr()
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.cuda.is_available()} "
          f"({'cuda:0 → ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'BRAK'})")
    import os
    cpu_count = os.cpu_count()
    print(f"  CPU:      {cpu_count} rdzeni")

    if sys.version_info >= (3, 13):
        print()
        print("  ⚠️  Python 3.13 wykryty!")
        print("     Znany bug: persistent_workers=True może powodować")
        print("     deadlock DataLoadera → efektywnie num_workers=0")
        print("     NAPRAWA: ustaw persistent_workers=False w datamodule.py")

    print()

    # ── 2. Format danych ─────────────────────────────────────────────────
    print("[ 2 ] FORMAT DANYCH")
    hr()
    patches_dir = dataset_root / "patches"
    tif_files = list(patches_dir.rglob("*-SPECTRAL_IMAGE.TIF"))[:5]
    npy_files = list(patches_dir.rglob("*-DATA.npy"))[:5]
    print(f"  .TIF dostępne: {'TAK' if tif_files else 'NIE'}")
    print(f"  .npy dostępne: {'TAK' if npy_files else 'NIE'}")

    if not npy_files:
        print()
        print("  ⚠️  Brak plików .npy!")
        print("     Uruchom: python scripts/convert_tif_to_npy.py <dataset_root>")
        print("     To 10-40x szybszy loader.")

    if npy_files:
        # Zmierz czas wczytywania .npy
        t0 = time.perf_counter()
        for f in npy_files:
            _ = np.load(str(f))
        t1 = time.perf_counter()
        ms_npy = (t1 - t0) / len(npy_files) * 1000
        print(f"  .npy load time: {ms_npy:.1f} ms/plik")
        if ms_npy > 20:
            print(f"  ⚠️  Wolne .npy (>{20}ms) → prawdopodobnie HDD zamiast SSD")

    if tif_files:
        t0 = time.perf_counter()
        import tifffile
        for f in tif_files:
            _ = tifffile.imread(str(f))
        t1 = time.perf_counter()
        ms_tif = (t1 - t0) / len(tif_files) * 1000
        print(f"  .TIF load time: {ms_tif:.1f} ms/plik")

    print()

    # ── 3. Dysk ──────────────────────────────────────────────────────────
    print("[ 3 ] DYSK")
    hr()
    import shutil
    stat = shutil.disk_usage(dataset_root)
    print(f"  Dostępne miejsce: {stat.free / 1e9:.1f} GB")

    # Sequential read test
    if npy_files:
        test_file = npy_files[0]
        file_size = test_file.stat().st_size / 1e6
        t0 = time.perf_counter()
        for _ in range(10):
            with open(test_file, "rb") as f:
                _ = f.read()
        t1 = time.perf_counter()
        read_speed = file_size * 10 / (t1 - t0)
        print(f"  Prędkość odczytu: {read_speed:.0f} MB/s")
        if read_speed < 200:
            print("  ⚠️  Wolny dysk (<200 MB/s) → prawdopodobnie HDD")
            print("     Rozważ przeniesienie datasetu na SSD/NVMe")
        elif read_speed < 1000:
            print("  (SATA SSD)")
        else:
            print("  (NVMe SSD) ✓")

    print()

    # ── 4. DataLoader (workers=0) ────────────────────────────────────────
    print("[ 4 ] DATALOADER — num_workers=0 (baseline)")
    hr()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from hsi_compression.utils import load_project_env
        load_project_env()
        from hsi_compression.data import build_dataloader, build_dataset

        ds = build_dataset(
            dataset_root=dataset_root,
            split_name="train",
            difficulty="easy",
            normalized=True,
            prefer_npy=bool(npy_files),
        )

        loader_w0 = build_dataloader(
            ds, batch_size=args.batch_size,
            shuffle=False, num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        n = min(args.batches, len(loader_w0))
        t0 = time.perf_counter()
        for i, batch in enumerate(loader_w0):
            if i >= n:
                break
            _ = batch["x"]
        t1 = time.perf_counter()
        ms_w0 = (t1 - t0) / n * 1000
        print(f"  {ms_w0:.0f} ms/batch  ({args.batch_size} patchy/batch, {n} batchy)")

    except Exception as e:
        print(f"  BŁĄD: {e}")
        ms_w0 = None

    print()

    # ── 5. DataLoader (workers=4,8) ──────────────────────────────────────
    for nw in [4, 8]:
        print(f"[ 5 ] DATALOADER — num_workers={nw}")
        hr()
        try:
            loader_wN = build_dataloader(
                ds, batch_size=args.batch_size,
                shuffle=False, num_workers=nw,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=False,   # False dla Python 3.13
            )

            # Warmup
            for i, batch in enumerate(loader_wN):
                if i >= 3:
                    break

            n = min(args.batches, len(loader_wN))
            t0 = time.perf_counter()
            for i, batch in enumerate(loader_wN):
                if i >= n:
                    break
                _ = batch["x"]
            t1 = time.perf_counter()
            ms_wN = (t1 - t0) / n * 1000
            speedup = (ms_w0 / ms_wN) if ms_w0 else 1.0
            print(f"  {ms_wN:.0f} ms/batch  ({speedup:.1f}x vs workers=0)")

        except Exception as e:
            print(f"  BŁĄD: {e}")

        print()

    # ── 6. GPU forward pass ──────────────────────────────────────────────
    print("[ 6 ] GPU FORWARD PASS (czyste obliczenia bez I/O)")
    hr()
    if not torch.cuda.is_available():
        print("  BRAK CUDA — trening będzie bardzo wolny na CPU!")
    else:
        try:
            from hsi_compression.models.registry import build_model
            device = torch.device("cuda")
            model = build_model(
                "baseline_2d_ae",
                in_channels=202,
                latent_channels=8,
                hidden_channels=(128, 64),
            ).to(device)

            dummy = torch.randn(args.batch_size, 202, 128, 128, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(dummy)
            torch.cuda.synchronize()

            # Pomiar
            n = 20
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n):
                    _ = model(dummy)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            ms_gpu = (t1 - t0) / n * 1000
            print(f"  Forward pass: {ms_gpu:.1f} ms/batch")
            print(f"  Throughput:   {args.batch_size * 1000 / ms_gpu:.0f} patchy/s")

        except Exception as e:
            print(f"  BŁĄD: {e}")

    print()

    # ── 7. Pełna iteracja treningowa ─────────────────────────────────────
    print("[ 7 ] PEŁNA ITERACJA (I/O + forward + backward + optimizer)")
    hr()
    if torch.cuda.is_available():
        try:
            import torch.nn.functional as F
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            loader_bench = build_dataloader(
                ds, batch_size=args.batch_size,
                shuffle=False, num_workers=8,
                pin_memory=True,
                persistent_workers=False,
            )

            n = min(args.batches, len(loader_bench))
            times = []
            for i, batch in enumerate(loader_bench):
                if i >= n:
                    break
                t0 = time.perf_counter()
                x = batch["x"].to(device, non_blocking=True)
                mask = batch["valid_mask"].to(device, non_blocking=True)
                optimizer.zero_grad()
                out = model(x)
                loss = F.mse_loss(out["x_hat"], x)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            # Pomiń pierwsze 3 (warmup)
            times = times[3:]
            if times:
                avg = sum(times) / len(times)
                iters_per_epoch = 8037 / args.batch_size
                epoch_min = iters_per_epoch * avg / 1000 / 60
                print(f"  {avg:.0f} ms/iter  (avg z {len(times)} iteracji)")
                print(f"  Szacowany czas epoki: {epoch_min:.1f} min")
                print(f"  Szacowany czas 500 epok: {epoch_min*500/60:.1f} h")

        except Exception as e:
            print(f"  BŁĄD: {e}")

    print()
    print("=" * 55)
    print("  DIAGNOZA ZAKOŃCZONA")
    print("=" * 55)


if __name__ == "__main__":
    main()
