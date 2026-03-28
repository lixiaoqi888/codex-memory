import hashlib
import os
import plistlib
import shutil
import subprocess
import sys

from codex_memory.codex_data import default_codex_home


def default_emit_dir(codex_home=None, cwd=None):
    codex_home = default_codex_home(codex_home)
    token = "global"
    if cwd:
        token = hashlib.sha1(cwd.encode("utf-8")).hexdigest()[:10]
    return os.path.join(codex_home, "memory", "hook-runtime", token)


def default_autostart_db_path(codex_home=None):
    codex_home = default_codex_home(codex_home)
    return os.path.join(codex_home, "memory", "codex-memory.sqlite")


def default_launch_agents_dir():
    return os.path.expanduser("~/Library/LaunchAgents")


def launchd_label_for_cwd(cwd):
    normalized = os.path.abspath(cwd or os.getcwd())
    suffix = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return "com.openai.codex-memory.watch.{}".format(suffix)


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stage_runtime_bundle(emit_dir, source_root=None):
    source_root = source_root or project_root()
    bundle_root = os.path.join(emit_dir, "bundle")
    source_package = os.path.join(source_root, "codex_memory")
    target_package = os.path.join(bundle_root, "codex_memory")
    if os.path.exists(target_package):
        shutil.rmtree(target_package)
    os.makedirs(bundle_root, exist_ok=True)
    shutil.copytree(source_package, target_package)
    return bundle_root


def runtime_vendor_root(emit_dir):
    return os.path.join(emit_dir, "vendor")


def build_pythonpath(bundle_root, vendor_root=None):
    parts = [bundle_root]
    if vendor_root:
        parts.append(vendor_root)
    return os.pathsep.join(parts)


def bootstrap_runtime_vendor(emit_dir, source_root=None, pip_runner=None, packages=None):
    vendor_root = runtime_vendor_root(emit_dir)
    os.makedirs(vendor_root, exist_ok=True)
    packages = list(packages or ("qdrant-client>=1.16,<2", "fastembed>=0.7,<1"))
    if not packages:
        return vendor_root
    source_root = source_root or project_root()
    if pip_runner:
        pip_runner(vendor_root, packages, source_root)
        return vendor_root
    pip_binary = os.path.join(source_root, ".venv", "bin", "pip")
    if os.path.exists(pip_binary):
        command = [pip_binary]
    else:
        command = [sys.executable, "-m", "pip"]
    command.extend(
        [
            "install",
            "--upgrade",
            "--target",
            vendor_root,
            "--disable-pip-version-check",
        ]
    )
    command.extend(packages)
    subprocess.run(command, check=True)
    return vendor_root


def reset_runtime_logs(emit_dir):
    logs_dir = os.path.join(emit_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    for name in ("watch.stdout.log", "watch.stderr.log"):
        with open(os.path.join(logs_dir, name), "w", encoding="utf-8"):
            pass


def build_watch_command(
    cwd,
    bundle_root,
    emit_dir=None,
    poll_interval=2.0,
    limit=3,
    codex_home=None,
    db_path=None,
    emit_shutdown_event=True,
):
    command = [
        "/usr/bin/python3",
        "-m",
        "codex_memory",
        "watch",
        "--cwd",
        os.path.abspath(cwd),
        "--poll-interval",
        str(poll_interval),
        "--limit",
        str(limit),
    ]
    if emit_dir:
        command.extend(["--emit-dir", emit_dir])
    if codex_home:
        command.extend(["--codex-home", default_codex_home(codex_home)])
    if db_path:
        command.extend(["--db", db_path])
    if emit_shutdown_event:
        command.append("--emit-session-end-on-exit")
    return command


def build_launchd_plist(
    cwd,
    bundle_root,
    vendor_root=None,
    emit_dir=None,
    poll_interval=2.0,
    limit=3,
    codex_home=None,
    db_path=None,
    label=None,
    emit_shutdown_event=True,
):
    emit_dir = emit_dir or default_emit_dir(codex_home=codex_home, cwd=cwd)
    label = label or launchd_label_for_cwd(cwd)
    logs_dir = os.path.join(emit_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    codex_home = default_codex_home(codex_home)
    db_path = db_path or default_autostart_db_path(codex_home)
    return {
        "Label": label,
        "ProgramArguments": build_watch_command(
            cwd=cwd,
            bundle_root=bundle_root,
            emit_dir=emit_dir,
            poll_interval=poll_interval,
            limit=limit,
            codex_home=codex_home,
            db_path=db_path,
            emit_shutdown_event=emit_shutdown_event,
        ),
        "RunAtLoad": True,
        "KeepAlive": True,
        "WorkingDirectory": emit_dir,
        "EnvironmentVariables": {
            "PYTHONPATH": build_pythonpath(bundle_root, vendor_root),
            "CODEX_HOME": codex_home,
            "CODEX_MEMORY_FASTEMBED_CACHE_DIR": os.path.join(codex_home, "memory", "fastembed-cache"),
        },
        "StandardOutPath": os.path.join(logs_dir, "watch.stdout.log"),
        "StandardErrorPath": os.path.join(logs_dir, "watch.stderr.log"),
        "ProcessType": "Background",
    }


def autostart_paths(cwd, launch_agents_dir=None, codex_home=None, emit_dir=None):
    label = launchd_label_for_cwd(cwd)
    launch_agents_dir = os.path.expanduser(launch_agents_dir or default_launch_agents_dir())
    emit_dir = emit_dir or default_emit_dir(codex_home=codex_home, cwd=cwd)
    plist_path = os.path.join(launch_agents_dir, "{}.plist".format(label))
    return {
        "label": label,
        "launch_agents_dir": launch_agents_dir,
        "plist_path": plist_path,
        "emit_dir": emit_dir,
    }


def install_autostart(
    cwd,
    repo_root=None,
    emit_dir=None,
    launch_agents_dir=None,
    poll_interval=2.0,
    limit=3,
    codex_home=None,
    db_path=None,
    label=None,
    load=False,
    launchctl_runner=None,
    bootstrap_runtime=True,
    bootstrapper=None,
):
    cwd = os.path.abspath(cwd or os.getcwd())
    paths = autostart_paths(cwd, launch_agents_dir=launch_agents_dir, codex_home=codex_home, emit_dir=emit_dir)
    if label:
        paths["label"] = label
        paths["plist_path"] = os.path.join(paths["launch_agents_dir"], "{}.plist".format(label))
    os.makedirs(paths["launch_agents_dir"], exist_ok=True)
    os.makedirs(paths["emit_dir"], exist_ok=True)
    codex_home = default_codex_home(codex_home)
    bundle_root = stage_runtime_bundle(paths["emit_dir"], source_root=repo_root)
    vendor_root = None
    if bootstrapper:
        vendor_root = bootstrapper(paths["emit_dir"], source_root=repo_root)
    elif bootstrap_runtime:
        vendor_root = bootstrap_runtime_vendor(paths["emit_dir"], source_root=repo_root)
    reset_runtime_logs(paths["emit_dir"])
    plist_payload = build_launchd_plist(
        cwd=cwd,
        bundle_root=bundle_root,
        vendor_root=vendor_root,
        emit_dir=paths["emit_dir"],
        poll_interval=poll_interval,
        limit=limit,
        codex_home=codex_home,
        db_path=db_path or default_autostart_db_path(codex_home),
        label=paths["label"],
        emit_shutdown_event=True,
    )
    with open(paths["plist_path"], "wb") as handle:
        plistlib.dump(plist_payload, handle)

    loaded = False
    if load:
        runner = launchctl_runner or _run_launchctl
        runner(["bootout", "gui/{}".format(os.getuid()), paths["plist_path"]], check=False)
        runner(["bootstrap", "gui/{}".format(os.getuid()), paths["plist_path"]], check=True)
        loaded = True

    return {
        "cwd": cwd,
        "label": paths["label"],
        "plist_path": paths["plist_path"],
        "emit_dir": paths["emit_dir"],
        "bundle_root": bundle_root,
        "vendor_root": vendor_root,
        "installed": True,
        "loaded": loaded,
    }


def remove_autostart(cwd, launch_agents_dir=None, unload=False, launchctl_runner=None):
    cwd = os.path.abspath(cwd or os.getcwd())
    paths = autostart_paths(cwd, launch_agents_dir=launch_agents_dir)
    unloaded = False
    if unload and os.path.exists(paths["plist_path"]):
        runner = launchctl_runner or _run_launchctl
        runner(["bootout", "gui/{}".format(os.getuid()), paths["plist_path"]], check=False)
        unloaded = True
    removed = False
    if os.path.exists(paths["plist_path"]):
        os.remove(paths["plist_path"])
        removed = True
    return {
        "cwd": cwd,
        "label": paths["label"],
        "plist_path": paths["plist_path"],
        "installed": False,
        "removed": removed,
        "loaded": False,
        "unloaded": unloaded,
    }


def autostart_status(cwd, launch_agents_dir=None, codex_home=None, emit_dir=None):
    cwd = os.path.abspath(cwd or os.getcwd())
    paths = autostart_paths(cwd, launch_agents_dir=launch_agents_dir, codex_home=codex_home, emit_dir=emit_dir)
    return {
        "cwd": cwd,
        "label": paths["label"],
        "plist_path": paths["plist_path"],
        "emit_dir": paths["emit_dir"],
        "installed": os.path.exists(paths["plist_path"]),
    }


def _run_launchctl(arguments, check):
    command = ["launchctl"] + list(arguments)
    return subprocess.run(command, check=check, capture_output=True, text=True)
