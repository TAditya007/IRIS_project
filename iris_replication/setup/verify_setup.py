# ============================================================
# verify_setup.py
# ------------------------------------------------------------
# End-to-end verification of the iRIS replication environment.
# Checks:
#   1. Python version (3.10.x)
#   2. Required packages import + pinned versions
#   3. CUDA availability on RTX 4060
#   4. Project folder structure (from create_project.bat)
#   5. Core math sanity (Eq. 14-18 quantization, Eq. 12 JFI)
#   6. GPU matmul smoke test
#
# Usage:
#   conda activate iris
#   python setup\verify_setup.py
# ============================================================

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# ---- ANSI colors (Windows 10+ terminals support these) ----
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    fatal: bool = False


@dataclass
class Verifier:
    results: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "", fatal: bool = False):
        self.results.append(CheckResult(name, passed, detail, fatal))
        icon = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
        print(f"  {icon} {name}")
        if detail:
            print(f"         {detail}")

    def summary(self) -> int:
        total = len(self.results)
        passed = sum(r.passed for r in self.results)
        fatal = [r for r in self.results if not r.passed and r.fatal]
        print()
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  SUMMARY{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"  Passed: {passed}/{total}")
        if fatal:
            print(f"  {RED}Fatal failures: {len(fatal)}{RESET}")
            for r in fatal:
                print(f"    - {r.name}: {r.detail}")
            print(f"\n{RED}{BOLD}  VERDICT: SETUP INCOMPLETE{RESET}")
            return 2
        if passed == total:
            print(f"\n{GREEN}{BOLD}  VERDICT: ALL GREEN -- READY FOR PHASE 0{RESET}")
            return 0
        print(f"\n{YELLOW}{BOLD}  VERDICT: SETUP OK WITH WARNINGS{RESET}")
        return 1


def hr(title: str):
    print()
    print(f"{CYAN}{BOLD}{'-'*60}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'-'*60}{RESET}")


# ---------- 1. Python version ----------
def check_python(v: Verifier):
    hr("1. Python interpreter")
    ver = sys.version_info
    ok = (ver.major == 3 and ver.minor == 10)
    v.add(
        "Python 3.10.x",
        ok,
        f"found {ver.major}.{ver.minor}.{ver.micro} at {sys.executable}",
        fatal=not ok,
    )
    v.add("Platform", True, f"{platform.system()} {platform.release()} ({platform.machine()})")


# ---------- 2. Packages ----------
REQUIRED = {
    "torch":       "2.3.1",
    "torchvision": "0.18.1",
    "numpy":       "1.26.4",
    "scipy":       "1.13.1",
    "matplotlib":  "3.8.4",
    "yaml":        "6.0.1",
    "tqdm":        "4.66.4",
    "pandas":      "2.2.2",
}


def check_packages(v: Verifier):
    hr("2. Required packages")
    for mod, want in REQUIRED.items():
        try:
            m = importlib.import_module(mod)
            got = getattr(m, "__version__", "unknown")
            ok = got.startswith(want)
            v.add(
                f"{mod} == {want}",
                ok,
                f"installed: {got}" + ("" if ok else f"  {YELLOW}(version drift){RESET}"),
                fatal=False,
            )
        except ImportError as e:
            v.add(f"{mod} == {want}", False, f"import error: {e}", fatal=True)


# ---------- 3. CUDA ----------
def check_cuda(v: Verifier):
    hr("3. CUDA / GPU")
    import torch

    ok_cuda = torch.cuda.is_available()
    v.add("torch.cuda.is_available()", ok_cuda,
          "GPU detected" if ok_cuda else "no CUDA device -- check NVIDIA driver >= 525",
          fatal=not ok_cuda)

    if ok_cuda:
        name = torch.cuda.get_device_name(0)
        v.add("GPU device", True, name)
        cap = torch.cuda.get_device_capability(0)
        v.add("Compute capability", cap >= (8, 6), f"sm_{cap[0]}{cap[1]}", fatal=False)

        # GPU matmul smoke test
        try:
            a = torch.randn(2048, 2048, device="cuda")
            b = torch.randn(2048, 2048, device="cuda")
            c = a @ b
            torch.cuda.synchronize()
            v.add("GPU matmul 2048x2048", True, f"result sum={c.sum().item():.2f}")
            del a, b, c
            torch.cuda.empty_cache()
        except Exception as e:
            v.add("GPU matmul 2048x2048", False, str(e), fatal=True)


# ---------- 4. Project structure ----------
EXPECTED_DIRS = [
    "configs", "environment", "agent", "analysis",
    "checkpoints", "logs", "results", "results/figures", "data",
]
EXPECTED_FILES = [
    "configs/default.yaml",
    "environment/__init__.py",
    "environment/channel.py",
    "environment/allocation.py",
    "environment/ris_env.py",
    "environment/quantization.py",
    "agent/__init__.py",
    "agent/replay_buffer.py",
    "agent/dqn.py",
    "analysis/__init__.py",
    "analysis/says_fit.py",
    "analysis/quantization.py",
    "analysis/fairness.py",
    "train.py", "evaluate.py", "requirements.txt", "README.md",
    "commands.txt", "timeline.txt", "work done.txt",
]


def check_structure(v: Verifier):
    hr("4. Project structure")
    # setup/ lives one level above the project
    here = Path(__file__).resolve().parent
    proj = here.parent / "iris_replication"

    if not proj.exists():
        v.add("iris_replication/ exists", False,
              f"not found at {proj} -- run setup\\create_project.bat first",
              fatal=True)
        return

    v.add("iris_replication/ exists", True, str(proj))

    missing_dirs = [d for d in EXPECTED_DIRS if not (proj / d).is_dir()]
    v.add(f"All {len(EXPECTED_DIRS)} subfolders present",
          not missing_dirs,
          "missing: " + ", ".join(missing_dirs) if missing_dirs else "all present",
          fatal=bool(missing_dirs))

    missing_files = [f for f in EXPECTED_FILES if not (proj / f).is_file()]
    v.add(f"All {len(EXPECTED_FILES)} placeholder files present",
          not missing_files,
          "missing: " + ", ".join(missing_files) if missing_files else "all present",
          fatal=bool(missing_files))


# ---------- 5. Math sanity ----------
def check_math(v: Verifier):
    hr("5. Core math sanity (paper equations)")

    import numpy as np

    # --- Eq. 14-16: phase quantization ---
    b = 3
    L = 2 ** b          # Eq. 14
    Delta = np.pi / 2 ** b  # Eq. 15
    var_eps = Delta ** 2 / 12  # from text after Eq. 16
    expected_L = 8
    expected_Delta = np.pi / 8
    expected_var = (np.pi / 8) ** 2 / 12
    v.add("Eq.14  L = 2^b (b=3 -> 8)", L == expected_L, f"L={L}")
    v.add("Eq.15  Delta = pi/2^b", np.isclose(Delta, expected_Delta),
          f"Delta={Delta:.6f}, expected={expected_Delta:.6f}")
    v.add("Eq.16  sigma_eps^2 = Delta^2/12",
          np.isclose(var_eps, expected_var),
          f"var={var_eps:.6e}, expected={expected_var:.6e}")

    # --- Eq. 17: normalized array gain (small M form) ---
    def G_small_M(M: int, b: int) -> float:
        Delta = np.pi / 2 ** b
        Phi = np.sinc(Delta / (2 * np.pi))  # sin(x)/x = sinc(x/pi)
        return 1.0 / M + (1.0 - 1.0 / M) * Phi ** 2

    M_test = 8
    g1 = G_small_M(M_test, 1)
    g6 = G_small_M(M_test, 6)
    v.add("Eq.17  G(b) monotonic in b", g6 > g1,
          f"G(1)={g1:.4f}, G(6)={g6:.4f}")
    v.add("Eq.17  0 < G(b) <= 1", 0 < g1 < 1 and 0 < g6 <= 1,
          f"G(1)={g1:.4f}, G(6)={g6:.4f}")

    # --- Eq. 18: gamma^(Q) = G(b) * gamma ---
    gamma_ideal = 100.0
    gamma_q = g1 * gamma_ideal
    v.add("Eq.18  gamma^(Q) = G*gamma",
          gamma_q < gamma_ideal,
          f"gamma={gamma_ideal:.1f} -> gamma^(Q)={gamma_q:.2f} at b=1")

    # --- Eq. 12: Jain's Fairness Index ---
    R = np.array([1.0, 2.0, 3.0, 4.0])
    jfi = (R.sum() ** 2) / (len(R) * (R ** 2).sum())
    # perfect fairness check
    R_eq = np.array([2.0, 2.0, 2.0, 2.0])
    jfi_eq = (R_eq.sum() ** 2) / (len(R_eq) * (R_eq ** 2).sum())
    v.add("Eq.12  JFI in (0,1]", 0 < jfi <= 1, f"JFI(unequal)={jfi:.4f}")
    v.add("Eq.12  JFI == 1 for equal rates", np.isclose(jfi_eq, 1.0),
          f"JFI(equal)={jfi_eq:.4f}")

    # --- Eq. 13: SAYS pdf sanity with Table 1 params ---
    from scipy.special import erfc

    def says_pdf(x, lam, mu, sigma, a, bb, ap, bp):
        if x <= mu:
            return (lam / 2.0) * np.exp(lam * (2 * mu + lam * sigma**2 - 2 * a * x)) \
                   * erfc((mu + lam * sigma**2 - bb * x) / (sigma * np.sqrt(2)))
        else:
            return (lam / 2.0) * np.exp(lam * (2 * mu + lam * sigma**2 - 2 * ap * x)) \
                   * erfc((mu + lam * sigma**2 - bp * x) / (sigma * np.sqrt(2)))

    # Table 1
    params = dict(lam=0.000487, mu=9.033490, sigma=15.116251,
                  a=15.767309, bb=0.803189, ap=1.050565, bp=0.554320)
    xs = np.linspace(0.1, 40, 200)
    pdf_vals = np.array([says_pdf(x, **params) for x in xs])
    finite = np.all(np.isfinite(pdf_vals))
    nonneg = np.all(pdf_vals >= 0)
    v.add("Eq.13  SAYS pdf finite + non-negative",
          finite and nonneg,
          f"range=[{pdf_vals.min():.3e}, {pdf_vals.max():.3e}]")


# ---------- 6. Writer round-trip ----------
def check_io(v: Verifier):
    hr("6. File I/O round-trip")
    import tempfile
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("iRIS check\n")
            path = f.name
        with open(path) as f:
            content = f.read()
        os.remove(path)
        v.add("Write + read temp file", content == "iRIS check\n", "OK")
    except Exception as e:
        v.add("Write + read temp file", False, str(e), fatal=True)


# ---------- main ----------
def main() -> int:
    print()
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  iRIS Replication -- Setup Verifier{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    v = Verifier()
    check_python(v)
    check_packages(v)
    check_cuda(v)
    check_structure(v)
    check_math(v)
    check_io(v)
    return v.summary()


if __name__ == "__main__":
    sys.exit(main())

    