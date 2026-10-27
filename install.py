"""voice2memos installer.

Checks every dependency, installs whatever is missing, then checks again to
prove the install worked. Safe to rerun: anything already present is reported
as found and left alone.

    python install.py                      check, install what's missing, check again
    python install.py --check              report only; change nothing
    python install.py --cpu                skip the NVIDIA CUDA packages even if a GPU is found
    python install.py --whisper-model tiny pre-download a different Whisper size

Three stages run in order, and each must pass before the next starts:

1. System: Python 3.11+, tkinter, the venv module, Ollama installed and running.
2. Python packages: a .venv in the project folder holding requirements.txt,
   plus requirements-cuda.txt when an NVIDIA GPU with a CUDA 12+ driver is found.
3. Models: the Whisper model in the Hugging Face cache, and the Ollama model.

A script running on Python can't install Python, so a failed Python or tkinter
check prints the fix and stops. The bootstrap scripts (install.ps1 and
install.sh, build-order step 16) will close that gap.

Standard library only, because nothing else exists yet when this runs.
Windows is tested; the macOS and Linux branches follow the design doc but have
not been run.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
REQUIREMENTS_CUDA = ROOT / "requirements-cuda.txt"

MIN_PYTHON = (3, 11)
MIN_TK = 8.6  # tkinter's PhotoImage reads PNG only from Tk 8.6 on
MIN_CUDA = 12  # faster-whisper's GPU wheels need a driver that supports CUDA 12+
WHISPER_MODEL = "small"
OLLAMA_MODEL = "gemma4:e2b"

OLLAMA_WINDOWS_SETUP = "https://ollama.com/download/OllamaSetup.exe"
OLLAMA_LINUX_SCRIPT = "https://ollama.com/install.sh"
OLLAMA_DOWNLOAD_PAGE = "https://ollama.com/download"

SYSTEM = platform.system()  # "Windows", "Darwin" or "Linux"


# --- Terminal output ---------------------------------------------------------

BOLD, DIM, RED, GREEN, YELLOW, CYAN = "1", "2", "31", "32", "33", "36"
_color = False


def enable_color() -> bool:
    """Turn on ANSI colors when stdout is a terminal that can show them.

    NO_COLOR turns them off and FORCE_COLOR turns them on, per the usual conventions.
    """
    global _color
    if os.environ.get("NO_COLOR"):
        _color = False
    elif os.environ.get("FORCE_COLOR"):
        _color = True
    else:
        _color = sys.stdout.isatty() and _enable_windows_vt()
    return _color


def _enable_windows_vt() -> bool:
    """Windows 10 consoles print escape codes literally until VT processing is switched on."""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except (AttributeError, OSError):
        return False


def paint(text: str, *codes: str) -> str:
    if not _color or not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def say(text: str = "", *codes: str) -> None:
    print(paint(text, *codes), flush=True)


def good_to_go(message: str = "good to go") -> None:
    print(f"  {paint(message, CYAN)} {paint(':)', YELLOW, BOLD)}", flush=True)


def elapsed(start: float) -> str:
    seconds = time.monotonic() - start
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


# --- Checks and the stage runner ---------------------------------------------


@dataclass
class Check:
    """One line of the report.

    ``fix`` installs the missing piece and must itself be safe to call twice.
    ``fix=None`` on a failed check means it has to be fixed by hand, and
    ``hint`` says how. Checks of kind "info" or "warn" are context lines and
    never count as failures.
    """

    name: str
    ok: bool
    detail: str = ""
    fix: Callable[[], None] | None = None
    hint: str = ""
    kind: str = "check"


def note(name: str, detail: str, kind: str = "info") -> Check:
    return Check(name, True, detail, kind=kind)


class InstallError(Exception):
    """A fix step failed. The message tells the user what to do next."""


TAGS = {
    "ok": ("[ ok ]", GREEN),
    "missing": ("[miss]", RED),
    "info": ("[info]", DIM),
    "warn": ("[warn]", YELLOW),
}


def show(checks: list[Check]) -> None:
    width = max((len(c.name) for c in checks), default=0)
    for c in checks:
        tag = c.kind if c.kind != "check" else ("ok" if c.ok else "missing")
        label, color = TAGS[tag]
        print(f"  {paint(label, color)} {c.name:<{width}}  {paint(c.detail, DIM)}".rstrip(), flush=True)


def run_stage(title: str, check: Callable[[], list[Check]], *, install: bool) -> bool:
    """Check, install whatever is missing, then check again. True when everything passes."""
    say()
    say(title, BOLD)
    checks = check()
    show(checks)
    failed = [c for c in checks if not c.ok]
    if not failed:
        good_to_go()
        return True

    manual = [c for c in failed if c.fix is None]
    for c in manual:
        say(f"  {c.name}: {c.hint or 'has to be fixed by hand'}", YELLOW)
    if manual:
        return False
    if not install:
        say(f"  Missing: {', '.join(c.name for c in failed)}. Run python install.py to install.", YELLOW)
        return False

    # Several checks can share one fix (every package shares "pip install"), so run each fix once.
    fixes = list({id(c.fix): c.fix for c in failed}.values())
    try:
        for fix in fixes:
            fix()
    except InstallError as e:
        say(f"  {e}", RED)
        return False

    say("  Checking again...", DIM)
    checks = check()
    show(checks)
    still = [c for c in checks if not c.ok]
    if still:
        say(f"  Still missing after installing: {', '.join(c.name for c in still)}", RED)
        return False
    good_to_go()
    return True


# --- Running things ----------------------------------------------------------


def child_env(**extra: str) -> dict[str, str]:
    """Environment for child processes, with UTF-8 I/O so their output decodes on any code page."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    env.update(extra)
    return env


def capture(cmd: list[str], timeout: float = 60, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str] | None:
    """Run quietly and return the result, or None if it couldn't start or timed out."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env or child_env(),
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def show_command(cmd: list[str]) -> str:
    line = subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)
    return line.replace(str(ROOT) + os.sep, "")


def run_live(cmd: list[str], what: str, env: dict[str, str] | None = None, shown: str = "") -> None:
    """Run with output going straight to the terminal, so downloads show their own progress.

    ``shown`` replaces the echoed command line when the real one is unreadable,
    such as an inline Python script.
    """
    say(f"  > {what}", BOLD)
    say(f"    {shown or show_command(cmd)}", DIM)
    start = time.monotonic()
    try:
        code = subprocess.call(cmd, cwd=ROOT, env=env or child_env())
    except OSError as e:
        raise InstallError(f"{what} could not start: {e}") from e
    if code != 0:
        raise InstallError(f"{what} failed with exit code {code}. The output above says why; rerun install.py once it's sorted.")
    say(f"    done in {elapsed(start)}", DIM)


def download(url: str, dest: Path) -> None:
    """Fetch url into dest, printing progress as it goes."""
    say(f"  > Downloading {url}", BOLD)
    request = urllib.request.Request(url, headers={"User-Agent": "voice2memos-installer"})
    partial = dest.with_name(dest.name + ".part")
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30) as response, open(partial, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            last_shown = 0.0
            while chunk := response.read(1 << 18):
                out.write(chunk)
                done += len(chunk)
                if time.monotonic() - last_shown > 0.25:
                    _progress(done, total)
                    last_shown = time.monotonic()
        _progress(done, total, final=True)
        os.replace(partial, dest)
    except OSError as e:
        partial.unlink(missing_ok=True)
        raise InstallError(f"Downloading {url} failed: {e}") from e
    say(f"    done in {elapsed(start)}", DIM)


def _progress(done: int, total: int, final: bool = False) -> None:
    text = f"    {done / 1e6:,.1f} MB"
    if total:
        text += f" of {total / 1e6:,.1f} MB  ({done * 100 // total}%)"
    if sys.stdout.isatty():
        print("\r" + text.ljust(48), end="\n" if final else "", flush=True)
    elif final:
        print(text, flush=True)


# --- Stage 1: system ---------------------------------------------------------


def _hint(windows: str, mac: str, linux: str) -> str:
    return {"Windows": windows, "Darwin": mac}.get(SYSTEM, linux)


def check_system() -> list[Check]:
    return [check_python(), check_tkinter(), check_venv_module(), check_ollama_installed(), check_ollama_server()]


def check_python() -> Check:
    version = platform.python_version()
    if sys.version_info >= MIN_PYTHON:
        return Check("Python", True, f"{version} ({sys.executable})")
    want = ".".join(map(str, MIN_PYTHON))
    return Check(
        "Python",
        False,
        f"{version} is older than {want}",
        hint=_hint(
            f"install it with `winget install Python.Python.{want}`, then run install.py with that Python",
            f"install it with `brew install python@{want} python-tk@{want}`, then run install.py with python{want}",
            f"install python{want}, python{want}-tk and python{want}-venv with your package manager, then rerun",
        ),
    )


def check_tkinter() -> Check:
    minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    hint = _hint(
        "rerun the python.org installer, choose Modify, and tick 'tcl/tk and IDLE'",
        f"install it with `brew install python-tk@{minor}`",
        f"install it with `sudo apt install python{minor}-tk` or your distro's equivalent",
    )
    try:
        import tkinter

        tkinter.Tcl()  # loads the Tcl/Tk libraries without opening a window
    except Exception as e:  # ImportError, or TclError when the libraries are missing
        return Check("tkinter", False, f"{type(e).__name__}: {e}", hint=hint)
    if tkinter.TkVersion < MIN_TK:
        return Check("tkinter", False, f"Tk {tkinter.TkVersion} is older than {MIN_TK}", hint=hint)
    return Check("tkinter", True, f"Tk {tkinter.TkVersion}")


def check_venv_module() -> Check:
    if importlib.util.find_spec("venv") and importlib.util.find_spec("ensurepip"):
        return Check("venv module", True, "venv and ensurepip present")
    minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    return Check(
        "venv module",
        False,
        "ensurepip is missing, so a virtual environment can't be made",
        hint=f"install it with `sudo apt install python{minor}-venv` or your distro's equivalent",
    )


def find_ollama() -> str | None:
    """Path to the ollama executable, including the usual install spots a stale PATH might miss."""
    found = shutil.which("ollama")
    if found:
        return found
    candidates: list[Path] = []
    if SYSTEM == "Windows" and os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Ollama" / "ollama.exe")
    elif SYSTEM == "Darwin":
        candidates += [
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            Path("/opt/homebrew/bin/ollama"),
            Path("/usr/local/bin/ollama"),
        ]
    else:
        candidates += [Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")]
    return next((str(c) for c in candidates if c.is_file()), None)


def ollama_version(exe: str) -> str:
    result = capture([exe, "--version"], timeout=15)
    match = re.search(r"\d+\.\d+\.\d+", (result.stdout + result.stderr) if result else "")
    return match.group() if match else ""


def ollama_url() -> str:
    """Where the Ollama server listens, honoring OLLAMA_HOST the way Ollama itself does."""
    raw = os.environ.get("OLLAMA_HOST", "").strip()
    scheme, sep, rest = raw.partition("://")
    if sep:
        port = 443 if scheme == "https" else 80
    else:
        scheme, rest, port = "http", raw, 11434
    parsed = urllib.parse.urlsplit(f"{scheme}://{rest or '127.0.0.1'}")
    host = parsed.hostname or "127.0.0.1"
    if host in ("0.0.0.0", "::"):  # listen-on-everything addresses; connect locally instead
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parsed.port or port
    except ValueError:
        pass
    return f"{scheme}://{host}:{port}"


_no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def ollama_models(timeout: float = 2.0) -> list[str] | None:
    """Names of the pulled models, or None when the server isn't answering."""
    try:
        with _no_proxy.open(ollama_url() + "/api/tags", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except (OSError, ValueError, AttributeError):
        return None


def has_model(pulled: list[str], model: str) -> bool:
    """Ollama treats a name without a tag as name:latest."""

    def tagged(name: str) -> str:
        return name if ":" in name else name + ":latest"

    return tagged(model) in {tagged(n) for n in pulled}


def check_ollama_installed() -> Check:
    exe = find_ollama()
    if exe:
        return Check("Ollama", True, " ".join(filter(None, [ollama_version(exe), f"({exe})"])))
    can_install = (
        SYSTEM == "Windows"
        or (SYSTEM == "Darwin" and shutil.which("brew"))
        or (SYSTEM == "Linux" and shutil.which("curl"))
    )
    return Check(
        "Ollama",
        False,
        "not installed",
        fix=install_ollama if can_install else None,
        hint=f"download it from {OLLAMA_DOWNLOAD_PAGE}, then rerun install.py",
    )


def check_ollama_server() -> Check:
    url = ollama_url()
    if ollama_models() is not None:
        return Check("Ollama server", True, f"answering at {url}")
    return Check("Ollama server", False, f"nothing answering at {url}", fix=start_ollama)


def install_ollama() -> None:
    if find_ollama():
        return
    if SYSTEM == "Windows":
        setup = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
        download(OLLAMA_WINDOWS_SETUP, setup)
        try:
            # Inno Setup flags: a progress window, no questions, no reboot.
            run_live([str(setup), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], "Running the Ollama installer")
        finally:
            setup.unlink(missing_ok=True)
    elif SYSTEM == "Darwin":
        run_live(["brew", "install", "ollama"], "Installing Ollama with Homebrew")
    else:
        run_live(["sh", "-c", f"curl -fsSL {OLLAMA_LINUX_SCRIPT} | sh"], "Running Ollama's install script")
    if not find_ollama():
        raise InstallError("The Ollama installer finished but ollama still can't be found. Open a new terminal and rerun install.py.")


def start_ollama() -> None:
    if ollama_models() is not None:
        return
    exe = find_ollama()
    if not exe:
        raise InstallError("Ollama isn't installed, so it can't be started.")
    tray_app = Path(exe).with_name("ollama app.exe")
    if SYSTEM == "Windows" and tray_app.is_file():
        cmd = [str(tray_app)]  # the tray app runs the server, the way Ollama normally starts on Windows
    elif SYSTEM == "Darwin" and Path("/Applications/Ollama.app").exists():
        cmd = ["open", "-a", "Ollama"]
    else:
        cmd = [exe, "serve"]

    say("  > Starting Ollama", BOLD)
    say(f"    {show_command(cmd)}", DIM)
    start = time.monotonic()
    detach: dict = (
        {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    try:
        subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **detach)
    except OSError as e:
        raise InstallError(f"Ollama could not start: {e}") from e
    while time.monotonic() - start < 60:
        if ollama_models() is not None:
            say(f"    answering after {elapsed(start)}", DIM)
            return
        time.sleep(0.5)
    raise InstallError(f"Ollama was started but nothing answered at {ollama_url()} within a minute.")


# --- Stage 2: Python packages ------------------------------------------------


@dataclass
class Gpu:
    name: str = ""
    cuda: str = ""  # the highest CUDA version the driver supports, e.g. "13.4"

    @property
    def cuda_ok(self) -> bool:
        try:
            return bool(self.name) and int(self.cuda.split(".")[0]) >= MIN_CUDA
        except ValueError:
            return False


def parse_cuda_version(smi_output: str) -> str:
    """Most drivers print "CUDA Version: 12.4"; newer ones print "CUDA UMD Version: 13.4"."""
    match = re.search(r"CUDA\b[^:|\n]*?Version:\s*(\d+(?:\.\d+)?)", smi_output)
    return match.group(1) if match else ""


def detect_gpu() -> Gpu:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return Gpu()
    names = capture([smi, "--query-gpu=name", "--format=csv,noheader"], timeout=15)
    if not names or names.returncode != 0 or not names.stdout.strip():
        return Gpu()
    header = capture([smi], timeout=15)
    return Gpu(names.stdout.strip().splitlines()[0].strip(), parse_cuda_version(header.stdout if header else ""))


def gpu_note(gpu: Gpu, use_cuda: bool) -> Check:
    if not gpu.name:
        return note("GPU", "no NVIDIA GPU found, so Whisper will run on the CPU")
    if use_cuda:
        return note("GPU", f"{gpu.name} with a CUDA {gpu.cuda} driver, so the CUDA packages are included")
    if not gpu.cuda_ok:
        return note(
            "GPU",
            f"{gpu.name}, but its driver supports CUDA {gpu.cuda or '(unknown)'} and {MIN_CUDA}+ is needed. "
            "Using the CPU; update the NVIDIA driver to use the GPU.",
            kind="warn",
        )
    return note("GPU", f"{gpu.name}, but --cpu was given, so the CUDA packages are skipped")


def sync_service(path: Path) -> str:
    """Name of the cloud sync client whose folder holds path, or "" when there isn't one."""
    for part in path.parts:
        lower = part.lower()
        if lower.startswith("onedrive"):
            return "OneDrive"
        if lower == "dropbox":
            return "Dropbox"
        if lower in ("google drive", "my drive"):
            return "Google Drive"
        if lower == "mobile documents":
            return "iCloud Drive"
    return ""


def normalize(name: str) -> str:
    """Package names compare case-insensitively with runs of - _ . treated as equal (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


_REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def read_requirements(files: list[Path]) -> list[str]:
    """Top-level package names from requirements files, skipping comments and pip options."""
    names: list[str] = []
    seen: set[str] = set()
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            match = _REQUIREMENT_NAME.match(line)
            if match and normalize(match.group()) not in seen:
                seen.add(normalize(match.group()))
                names.append(match.group())
    return names


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def pip_list(python: Path) -> dict[str, str]:
    """Installed packages in the venv, as normalized name to version."""
    result = capture([str(python), "-m", "pip", "list", "--format=json", "--disable-pip-version-check"])
    if not result or result.returncode != 0:
        return {}
    try:
        return {normalize(p["name"]): p["version"] for p in json.loads(result.stdout)}
    except (ValueError, KeyError, TypeError):
        return {}


def pip_pending(python: Path, files: list[Path]) -> list[str]:
    """What pip would still install to satisfy the requirement files. Empty means all in place.

    This catches what a name-by-name check can't: a wrong version, or a
    dependency of a dependency that went missing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        cmd = [str(python), "-m", "pip", "install", "--dry-run", "--quiet", "--disable-pip-version-check", "--report", str(report)]
        for path in files:
            cmd += ["-r", str(path)]
        result = capture(cmd, timeout=300)
        try:
            if not result or result.returncode != 0:
                raise ValueError
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ["(pip couldn't resolve the requirements)"]
    return [f"{i['metadata']['name']} {i['metadata']['version']}" for i in data.get("install", [])]


def check_packages(files: list[Path], notes: list[Check]) -> list[Check]:
    fix = functools.partial(install_packages, files)
    checks = list(notes)
    names = read_requirements(files)
    python = venv_python()
    if not python.is_file():
        checks.append(Check("virtual env", False, f"no {VENV.name} folder yet", fix=fix))
        checks += [Check(name, False, "not installed", fix=fix) for name in names]
        return checks

    checks.append(Check("virtual env", True, VENV.name))
    installed = pip_list(python)
    for name in names:
        version = installed.get(normalize(name))
        checks.append(Check(name, bool(version), version or "not installed", fix=fix))
    if all(normalize(name) in installed for name in names):
        pending = pip_pending(python, files)
        detail = "needs " + ", ".join(pending) if pending else "everything they depend on is in place too"
        checks.append(Check("dependencies", not pending, detail, fix=fix))
    if SYSTEM == "Linux" and "sounddevice" in installed:
        checks.append(check_portaudio(python))
    return checks


def install_packages(files: list[Path]) -> None:
    python = venv_python()
    if not python.is_file():
        run_live([sys.executable, "-m", "venv", str(VENV)], f"Creating the virtual environment in {VENV.name}")
    cmd = [str(python), "-m", "pip", "install", "--disable-pip-version-check"]
    for path in files:
        cmd += ["-r", path.name]
    run_live(cmd, "Installing Python packages")


def check_portaudio(python: Path) -> Check:
    """Linux only: pip's sounddevice wheel there needs the system's libportaudio."""
    result = capture([str(python), "-c", "import sounddevice"])
    if result and result.returncode == 0:
        return Check("PortAudio", True, "sounddevice loads it")
    can_fix = shutil.which("apt-get") and shutil.which("sudo")
    return Check(
        "PortAudio",
        False,
        "sounddevice can't load libportaudio",
        fix=install_portaudio if can_fix else None,
        hint="install libportaudio2 (or portaudio) with your package manager",
    )


def install_portaudio() -> None:
    run_live(["sudo", "apt-get", "install", "-y", "libportaudio2"], "Installing PortAudio")


# --- Stage 3: models ---------------------------------------------------------

# Both run inside the venv, where faster-whisper is installed. faster-whisper
# hides the Hugging Face progress bar while downloading, so the download script
# swaps the bar back in. The model lands in the Hugging Face cache, which
# honors HF_HOME for anyone who wants it off the system drive.
_WHISPER_CACHED = """
import sys
from faster_whisper.utils import download_model
print(download_model(sys.argv[1], local_files_only=True))
"""

_WHISPER_DOWNLOAD = """
import sys
from faster_whisper import utils
from huggingface_hub.utils import tqdm
utils.disabled_tqdm = tqdm
print(utils.download_model(sys.argv[1]))
"""


def _whisper_env() -> dict[str, str]:
    return child_env(HF_HUB_DISABLE_SYMLINKS_WARNING="1")


def check_models(whisper_model: str, ollama_model: str) -> list[Check]:
    return [check_whisper_model(whisper_model), check_ollama_model(ollama_model)]


def check_whisper_model(size: str) -> Check:
    fix = functools.partial(download_whisper_model, size)
    python = venv_python()
    if not python.is_file():
        return Check("Whisper model", False, f"{size}: needs the Python packages first", fix=fix)
    result = capture([str(python), "-c", _WHISPER_CACHED, size], timeout=120, env=_whisper_env())
    if result and result.returncode == 0:
        return Check("Whisper model", True, f"{size}, downloaded")
    return Check("Whisper model", False, f"{size}, not downloaded yet", fix=fix)


def download_whisper_model(size: str) -> None:
    python = venv_python()
    run_live(
        [str(python), "-c", _WHISPER_DOWNLOAD, size],
        f"Downloading the Whisper {size} model",
        env=_whisper_env(),
        shown=f"{show_command([str(python)])} -c \"faster_whisper.utils.download_model('{size}')\"",
    )


def check_ollama_model(model: str) -> Check:
    fix = functools.partial(pull_ollama_model, model)
    pulled = ollama_models()
    if pulled is None:
        return Check("Ollama model", False, f"{model}: the Ollama server isn't answering", fix=fix)
    if has_model(pulled, model):
        return Check("Ollama model", True, f"{model}, pulled")
    return Check("Ollama model", False, f"{model}, not pulled yet", fix=fix)


def pull_ollama_model(model: str) -> None:
    start_ollama()
    exe = find_ollama()
    if not exe:
        raise InstallError("Ollama isn't installed, so the model can't be pulled.")
    run_live([exe, "pull", model], f"Pulling {model} (several GB, the biggest download here)")


# --- Main --------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and install everything voice2memos needs. Safe to rerun.")
    parser.add_argument("--check", action="store_true", help="report what's present and missing, and change nothing")
    parser.add_argument("--cpu", action="store_true", help="skip the NVIDIA CUDA packages even when a GPU is found")
    parser.add_argument(
        "--whisper-model", default=WHISPER_MODEL, metavar="SIZE", help=f"Whisper model to download (default {WHISPER_MODEL})"
    )
    parser.add_argument(
        "--ollama-model", default=OLLAMA_MODEL, metavar="NAME", help=f"Ollama model to pull (default {OLLAMA_MODEL})"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    enable_color()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    install = not args.check
    start = time.monotonic()
    say(f"voice2memos installer, {SYSTEM} {platform.machine()}" + (", check only" if args.check else ""), BOLD)

    try:
        # Each stage needs the one before it, so stop at the first failure.
        # A --check run reports on every stage anyway, since it changes nothing.
        ok = run_stage("[1/3] System", check_system, install=install)

        if ok or args.check:
            gpu = detect_gpu()
            use_cuda = gpu.cuda_ok and not args.cpu
            files = [REQUIREMENTS] + ([REQUIREMENTS_CUDA] if use_cuda else [])
            notes = [gpu_note(gpu, use_cuda)]
            if service := sync_service(ROOT):
                size = "over 2 GB with the CUDA packages" if use_cuda else "about 300 MB in 5,000 files"
                notes.append(
                    note(
                        "project folder",
                        f"inside {service}, which will also sync {VENV.name} ({size}). "
                        f"Cloning the repo outside {service} avoids that.",
                        kind="warn",
                    )
                )
            ok = run_stage("[2/3] Python packages", functools.partial(check_packages, files, notes), install=install) and ok

        if ok or args.check:
            models = functools.partial(check_models, args.whisper_model, args.ollama_model)
            ok = run_stage("[3/3] Models", models, install=install) and ok
    except KeyboardInterrupt:
        say()
        say("Stopped. Rerun install.py to pick up where it left off.", YELLOW)
        return 130

    say()
    if not ok:
        say("Not done yet. Fix what's listed above and rerun install.py; finished steps will just say found.", RED)
        return 1
    message = "You're good to go"
    print(f"{paint(message, CYAN, BOLD)} {paint(':)', YELLOW, BOLD)}  {paint(elapsed(start), DIM)}", flush=True)
    launch = r".venv\Scripts\python -m voice2memos" if os.name == "nt" else ".venv/bin/python -m voice2memos"
    say(f"Start the app from this folder with: {launch}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
