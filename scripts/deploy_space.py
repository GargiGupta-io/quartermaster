"""Publish the committed code to a Hugging Face Space and set its secrets.

Reads HF_TOKEN, SPACE_ID, GROQ_API_KEY and SPACE_API_KEY from .env. Only files committed to git
are sent, so .env, saved requests and the local index never leave this machine.

    uv run --with huggingface_hub python scripts/deploy_space.py --dry-run   # show, send nothing
    uv run --with huggingface_hub python scripts/deploy_space.py             # publish
"""

import argparse
import subprocess
import tarfile
import tempfile
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

# Hugging Face reads the Space's settings from the top of its README.
SPACE_HEADER = """---
title: Procurement Agent
emoji: 📦
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Procurement assistant with human approval
---

"""
SECRETS = ("GROQ_API_KEY", "API_KEY")


class DeploySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    hf_token: str = ""
    space_id: str = ""  # for example "your-name/procurement-agent"
    space_private: bool = False
    groq_api_key: str = ""
    space_api_key: str = ""  # the access key people will type into the deployed page


def stage_committed_files(target: Path) -> list[str]:
    """Copy what is committed at HEAD, nothing else, and add the Space header to the README."""
    archive = target / "head.tar"
    subprocess.run(["git", "archive", "--output", archive, "HEAD"], cwd=ROOT, check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(target / "space", filter="data")
    archive.unlink()
    space = target / "space"
    readme = space / "README.md"
    readme.write_text(SPACE_HEADER + readme.read_text(encoding="utf-8"), encoding="utf-8")
    return sorted(str(p.relative_to(space)) for p in space.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="show what would be sent")
    args = parser.parse_args()

    settings = DeploySettings()
    missing = [
        name
        for name, value in [
            ("HF_TOKEN", settings.hf_token),
            ("SPACE_ID", settings.space_id),
            ("GROQ_API_KEY", settings.groq_api_key),
            ("SPACE_API_KEY", settings.space_api_key),
        ]
        if not value
    ]
    if missing and not args.dry_run:
        raise SystemExit(f"Add these to .env first: {', '.join(missing)}")
    if missing:
        print(f"Still needed in .env before publishing: {', '.join(missing)}")

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        files = stage_committed_files(Path(tmp))
        visibility = "private" if settings.space_private else "public"
        print(
            f"Space: {settings.space_id or '(SPACE_ID not set)'} ({visibility}, Docker, free CPU)"
        )
        print(f"Code: commit {commit}, {len(files)} files:")
        for name in files:
            print(f"  {name}")
        print(f"Secrets set on the Space (values not shown): {', '.join(SECRETS)}")
        if args.dry_run:
            print("Dry run: nothing was sent.")
            return

        from huggingface_hub import HfApi

        api = HfApi(token=settings.hf_token)
        api.create_repo(
            settings.space_id,
            repo_type="space",
            space_sdk="docker",
            private=settings.space_private,
            exist_ok=True,
        )
        api.add_space_secret(settings.space_id, "GROQ_API_KEY", settings.groq_api_key)
        api.add_space_secret(settings.space_id, "API_KEY", settings.space_api_key)
        api.upload_folder(
            folder_path=Path(tmp) / "space",
            repo_id=settings.space_id,
            repo_type="space",
            commit_message=f"Deploy {commit}",
            delete_patterns=["*"],  # the Space mirrors the commit: removed files are removed
        )
        owner, name = settings.space_id.split("/")
        print(f"Published. Building now: https://huggingface.co/spaces/{settings.space_id}")
        print(f"App address once built: https://{owner}-{name}.hf.space".lower())


if __name__ == "__main__":
    main()
