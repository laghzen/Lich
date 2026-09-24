from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple


_MSVC_VERSION_RE = re.compile(r"^14\.(\d+)(?:\.|$)")


def _version_key(path: Path) -> Tuple[int, str]:
    m = _MSVC_VERSION_RE.match(path.name)
    if not m:
        return (-1, path.name)
    return (int(m.group(1)), path.name)


def _find_vs_installation() -> Optional[Path]:
    # Prefer an explicit VS environment supplied by a Developer Command Prompt.
    for key in ("VSINSTALLDIR",):
        raw = os.environ.get(key)
        if raw and Path(raw).exists():
            return Path(raw)

    # Standard Visual Studio Installer location for vswhere, which can discover
    # installs even when the product itself lives on a non-C: drive.
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
    ]
    for vswhere in candidates:
        if not vswhere.exists():
            continue
        try:
            out = subprocess.check_output(
                [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if out and Path(out).exists():
            return Path(out)
    return None


def _available_msvc_tools(vs_install: Path) -> list[Path]:
    root = vs_install / "VC" / "Tools" / "MSVC"
    if not root.exists():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "bin" / "Hostx64" / "x64" / "cl.exe").exists()],
        key=_version_key,
        reverse=True,
    )


def configure_windows_msvc(*, quiet: bool = False) -> Dict[str, str]:
    """Load a native MSVC x64 environment and force NVCC to use cl.exe.

    CUDA 12.9 documents MSVC 193x as the supported Windows host compiler family.
    Prefer an installed 14.39/14.38 toolset when available; otherwise fall back to
    the newest installed MSVC and report that it is outside the documented range.

    The environment is returned and also merged into os.environ for the current
    Python process. This is intentionally process-local; no system-wide variables
    are changed.
    """
    if os.name != "nt":
        return {}

    vs = _find_vs_installation()
    if vs is None:
        raise RuntimeError(
            "Visual Studio installation was not found. Run from a VS Developer Prompt "
            "or install Visual Studio 2022 C++ build tools."
        )

    tools = _available_msvc_tools(vs)
    if not tools:
        raise RuntimeError(
            f"No MSVC x64 toolset found under {vs / 'VC' / 'Tools' / 'MSVC'}. "
            "Install the VS 2022 C++ x64/x86 build tools."
        )

    # 14.38/14.39 map to the 193x MSVC family requested by CUDA 12.9 docs.
    preferred = [p for p in tools if p.name.startswith(("14.39.", "14.38."))]
    chosen = preferred[0] if preferred else tools[0]
    chosen_name = chosen.name
    chosen_minor = _version_key(chosen)[0]
    supported_family = 38 <= chosen_minor <= 39

    vcvars = vs / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        raise RuntimeError(f"Missing Visual Studio environment script: {vcvars}")

    # Import the full vcvars environment into this process. Do NOT pass the
    # quoted BAT path as an argv item to cmd.exe: Python's Windows argument
    # quoting can turn the embedded quotes into literal `\"` characters.
    # A temporary .cmd file is robust across cmd/Python combinations.
    import tempfile

    version_arg = chosen_name.rsplit(".", 1)[0]
    script_text = (
        "@echo off\n"
        f'call "{vcvars}" -vcvars_ver={version_arg}\n'
        "if errorlevel 1 exit /b %errorlevel%\n"
        "set\n"
    )
    def _run_capture_script(contents: str) -> str:
        with tempfile.TemporaryDirectory(prefix="ipc_msvc_") as td:
            env_script = Path(td) / "capture_env.cmd"
            env_script.write_text(contents, encoding="utf-8", newline="\r\n")
            # Execute from the temporary directory and pass only the filename.
            # This avoids cmd.exe's special /c parsing rules for quoted absolute
            # paths (especially paths containing spaces).
            return subprocess.check_output(
                ["cmd.exe", "/d", "/q", "/c", "capture_env.cmd"],
                cwd=td,
                text=True,
                stderr=subprocess.STDOUT,
                encoding="mbcs" if os.name == "nt" else None,
                errors="replace",
            )

    try:
        raw = _run_capture_script(script_text)
    except subprocess.CalledProcessError as exc:
        detail = exc.output.strip() if exc.output else ""
        # Some VS installations have the selected toolset directory but do not
        # expose that version through -vcvars_ver. Retry with the default x64
        # environment, then force the selected cl.exe to the front of PATH.
        fallback_text = (
            "@echo off\n"
            f'call "{vcvars}"\n'
            "if errorlevel 1 exit /b %errorlevel%\n"
            "set\n"
        )
        try:
            raw = _run_capture_script(fallback_text)
        except subprocess.CalledProcessError as exc2:
            detail2 = exc2.output.strip() if exc2.output else ""
            raise RuntimeError(
                "Failed to initialize MSVC environment.\n"
                f"Versioned vcvars attempt:\n{detail}\n"
                f"Default vcvars attempt:\n{detail2}"
            ) from exc2

    # `set` emits KEY=VALUE lines. Import the environment into this Python
    # process; simply running vcvars in a child process is not sufficient.
    # Ignore diagnostic lines emitted by vcvars and accept normal environment
    # variable names only.
    imported: Dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        imported[key] = value
        os.environ[key] = value

    cl = chosen / "bin" / "Hostx64" / "x64" / "cl.exe"
    if not cl.exists():
        raise RuntimeError(f"Selected MSVC compiler disappeared: {cl}")

    # Make the selected toolset win over any clang-cl or newer MSVC entry
    # inherited from the parent shell.
    selected_bin = str(cl.parent)
    path_parts = [x for x in os.environ.get("PATH", "").split(os.pathsep) if x]
    path_parts = [x for x in path_parts if Path(x).resolve() != cl.parent.resolve()]
    os.environ["PATH"] = selected_bin + os.pathsep + os.pathsep.join(path_parts)

    # These are read by nvcc/TileLang. In particular this prevents a global
    # CXX=clang-cl.exe setting from hijacking kernel compilation.
    os.environ["CC"] = "cl.exe"
    os.environ["CXX"] = "cl.exe"
    os.environ["NVCC_CCBIN"] = "cl.exe"
    os.environ["IPC_MSVC_TOOLSET"] = chosen_name
    os.environ["IPC_MSVC_ROOT"] = str(chosen)
    os.environ["IPC_MSVC_CL"] = str(cl)

    imported.update({
        "CC": "cl.exe",
        "CXX": "cl.exe",
        "NVCC_CCBIN": "cl.exe",
        "IPC_MSVC_TOOLSET": chosen_name,
        "IPC_MSVC_ROOT": str(chosen),
        "IPC_MSVC_CL": str(cl),
    })

    if not quiet:
        status = "supported CUDA 12.9 family (MSVC 193x)" if supported_family else "OUTSIDE CUDA 12.9 documented MSVC family (193x)"
        print(f"[windows-toolchain] VS: {vs}")
        print(f"[windows-toolchain] MSVC: {chosen_name} ({status})")
        print(f"[windows-toolchain] cl.exe: {cl}")
        print("[windows-toolchain] NVCC host compiler: cl.exe")
        if not supported_family:
            print("[windows-toolchain] WARNING: install MSVC v14.39-17.9 or v14.38-17.8 for the documented CUDA 12.9 host toolchain.")

    return imported
