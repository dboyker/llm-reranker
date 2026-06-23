"""Module d'installation des dependances sur Databricks.

Attention!:
La commande d'installation utilise le flag "--break-system-packages" afin de bypasser la gestion des
venvs de Databricks. Elle ne doit pas être utilisée en local!
"""
import os
import subprocess
import tomllib
from importlib.metadata import distributions

from packaging.requirements import Requirement

PYPROJECT = "../pyproject.toml"


def normalize_package_name(name: str) -> str:
    """Normalize a package name."""
    return name.replace("-", "_").lower()


def find_missing_dependencies(pyproject_dependencies: list[str]) -> list[str]:
    """A partir des deps spécifiées dans le pyproject, détermine et renvoi les deps manquantes."""
    wanted = {normalize_package_name(Requirement(dep).name) for dep in pyproject_dependencies}
    installed = {normalize_package_name(dist.metadata["Name"]) for dist in distributions()}
    missing = sorted(wanted - installed)
    print("Missing librairies: ", missing)
    return missing


def install() -> None:
    """Fonction d'installation."""
    # Safeguard pour éviter l'installation locale
    if (os.getenv("DATABRICKS_RUNTIME") is None) and (os.getenv("DATABRICKS_ROOT_VIRTUALENV_ENV") is None):
        msg = "Ce script ne doit pas être effectué en local!"
        raise RuntimeError(msg)
    
    # Project definition
    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)

    # Dependencies
    deps = data.get("project", {}).get("dependencies", [])
    if not deps:
        print("No [project.dependencies] found.")
        return
    
    # Find missing deps
    missing_deps = find_missing_dependencies(deps)
    if not missing_deps:
        print("All direct dependencies already installed.")
        return

    # Installation
    shell_commands = [
        ["pip", "install", "uv"],
        ["uv", "pip", "install", "--system", "--break-system-packages"] + missing_deps
    ]
    for cmd in shell_commands:
        print(cmd)
        subprocess.check_call(cmd)


if __name__ == "__main__":
    install()