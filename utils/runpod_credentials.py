"""RunPod credentials resolver.

Profile-based credentials lookup so multiple RunPod accounts can coexist
(work account + personal account, etc.). Mirrors AWS / gcloud's pattern
of an INI file with named sections.

Resolution order (first hit wins):
    1. profile arg  -> credentials.ini[<profile>]
    2. RUNPOD_PROFILE env var -> credentials.ini[<that name>]
    3. Bare RUNPOD_API_KEY + RUNPOD_USER_ID env vars (single-account
       fallback for first-time users who haven't set up the file yet)

Credentials file path:
    %USERPROFILE%/.runpod/credentials.ini  (Windows)
    ~/.runpod/credentials.ini              (POSIX)

File format:
    [default]
    api_key  = <key>
    user_id  = <user-id>
    label    = work     ; optional, cosmetic only

    [personal]
    api_key  = <other key>
    user_id  = <other user-id>
    label    = personal
"""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunpodCreds:
    """Resolved RunPod account credentials.

    `label` is purely cosmetic for log lines so the user can tell which
    account a dispatch ran under at a glance.
    """
    api_key: str
    user_id: str
    label: str = "default"


CREDS_FILE = Path.home() / ".runpod" / "credentials.ini"

# These are the env vars consulted as fallback. Documenting them here
# so the dispatcher's error message can name them precisely.
ENV_PROFILE = "RUNPOD_PROFILE"
ENV_API_KEY = "RUNPOD_API_KEY"
ENV_USER_ID = "RUNPOD_USER_ID"


class RunpodCredsError(RuntimeError):
    """Raised when no usable credentials can be resolved."""


def resolve(profile: str | None = None) -> RunpodCreds:
    """Resolve RunPod credentials per the documented precedence order.

    Args:
        profile: Optional profile name from the dispatcher node's widget.
            Empty string or None means "fall back to env / default".

    Returns:
        A RunpodCreds with both api_key and user_id non-empty.

    Raises:
        RunpodCredsError with a precise hint about how to fix it when
        nothing resolves.
    """
    profile = (profile or "").strip()

    # 1. Explicit profile arg from the node widget.
    if profile:
        creds = _from_file(profile)
        if creds is not None:
            return creds
        raise RunpodCredsError(
            f"Profile '{profile}' not found. Expected a section "
            f"[{profile}] in {CREDS_FILE} with api_key + user_id."
        )

    # 2. RUNPOD_PROFILE env var selects a section name.
    env_profile = os.environ.get(ENV_PROFILE, "").strip()
    if env_profile:
        creds = _from_file(env_profile)
        if creds is not None:
            return creds
        raise RunpodCredsError(
            f"{ENV_PROFILE}={env_profile!r} but no [{env_profile}] "
            f"section found in {CREDS_FILE}."
        )

    # 3. credentials.ini[default] if the file exists.
    creds = _from_file("default")
    if creds is not None:
        return creds

    # 4. Bare env vars — single-account fallback for first-time users.
    api_key = os.environ.get(ENV_API_KEY, "").strip()
    user_id = os.environ.get(ENV_USER_ID, "").strip()
    if api_key and user_id:
        return RunpodCreds(api_key=api_key, user_id=user_id, label="env")

    # Nothing worked — give the user a precise hint.
    raise RunpodCredsError(
        "No RunPod credentials found. Set up either:\n"
        f"  (a) {CREDS_FILE} with a [default] section containing "
        f"api_key and user_id, OR\n"
        f"  (b) {ENV_API_KEY} and {ENV_USER_ID} environment variables.\n"
        "Get your API key + account user_id from "
        "https://www.runpod.io/console/user/settings"
    )


def _from_file(section: str) -> RunpodCreds | None:
    """Load a single profile from the credentials file.

    Returns None if the file or section doesn't exist; raises if the
    section exists but is missing required fields (so the user gets a
    clear error rather than a silent fallback).
    """
    if not CREDS_FILE.exists():
        return None

    parser = configparser.ConfigParser()
    try:
        parser.read(CREDS_FILE, encoding="utf-8")
    except configparser.Error as e:
        raise RunpodCredsError(
            f"Could not parse {CREDS_FILE}: {e}"
        )

    if section not in parser:
        return None

    sect = parser[section]
    api_key = sect.get("api_key", "").strip()
    user_id = sect.get("user_id", "").strip()
    label = sect.get("label", section).strip() or section

    if not api_key:
        raise RunpodCredsError(
            f"{CREDS_FILE} [{section}] is missing required field "
            "'api_key'."
        )
    if not user_id:
        raise RunpodCredsError(
            f"{CREDS_FILE} [{section}] is missing required field "
            "'user_id'."
        )

    return RunpodCreds(api_key=api_key, user_id=user_id, label=label)
