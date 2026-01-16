import os
import sys
import json
import time
import random
import shutil
import subprocess
from datetime import datetime, timedelta
from dateutil import parser as dtparser

import requests

# ----------------------------
# CONFIG (tweak as needed)
# ----------------------------
TARGET_TOTAL_IMAGES = int(os.environ.get("TARGET_TOTAL_IMAGES", "210"))  # 50-100 recommended
OFFICIAL_RATIO = float(os.environ.get("OFFICIAL_RATIO", "0.5"))         # 0.5 => half official, half community

MIN_STARS = int(os.environ.get("MIN_STARS", "10"))
MIN_PULLS = int(os.environ.get("MIN_PULLS", "10000"))                   # 10k pulls
MAX_LAST_UPDATED_DAYS = int(os.environ.get("MAX_LAST_UPDATED_DAYS", "1000"))  # 1000 days

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data/scans")
DOCKER_CLI = os.environ.get("DOCKER_CLI", "docker")
TRIVY_CLI = os.environ.get("TRIVY_CLI", "trivy")

# ----------------------------
# API Endpoints
# ----------------------------
HUB_BASE = "https://hub.docker.com"
API_REPOS = f"{HUB_BASE}/v2/repositories/"
API_SEARCH = f"{HUB_BASE}/api/content/v1/products/search"  # better for verified/org filters

# ----------------------------
# Helpers
# ----------------------------
def check_cli_available(cmd):
    return shutil.which(cmd) is not None

def run_cmd(cmd, cwd=None):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err

def iso_to_dt(s):
    try:
        return dtparser.parse(s)
    except Exception:
        return None

def is_recent(dt, max_days):
    if not dt:
        return False
    from datetime import timezone
    return (datetime.now(timezone.utc) - dt) <= timedelta(days=max_days)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def pick_latest_tag(namespace, repo_name):
    """Return ('repo:tag', tag) preferring 'latest', else most recently updated tag."""
    tags_url = f"{API_REPOS}{namespace}/{repo_name}/tags/?page_size=50&ordering=last_updated"
    r = requests.get(tags_url, timeout=20)
    if r.status_code != 200:
        return None, None
    results = r.json().get("results", [])
    if not results:
        return None, None

    # Prefer 'latest'
    for t in results:
        if t.get("name") == "latest":
            return f"{namespace}/{repo_name}:latest", "latest"

    # else pick most recently updated tag
    best = results[0]
    return f"{namespace}/{repo_name}:{best.get('name')}", best.get('name')

def filter_candidates(items, want_official):
    """items come from Docker Hub APIs with varied fields. Apply safety filters."""
    filtered = []
    for it in items:
        # Normalize shape across endpoints
        namespace = it.get("namespace") or it.get("user") or it.get("publisher", {}).get("name")
        name = it.get("name") or it.get("repo_name")
        if not namespace or not name:
            full_name = it.get("slug")  # sometimes available
            if full_name and "/" in full_name:
                namespace, name = full_name.split("/", 1)
            else:
                continue

        # official flag
        is_official = it.get("is_official")
        if is_official is None:
            # Some search API returns 'is_official' under 'images' or product metadata
            is_official = it.get("official", False)

        if want_official and not is_official:
            continue
        if not want_official and is_official:
            continue

        # Verified / trusted publishers preferred for community
        is_verified = it.get("is_verified") or it.get("publisher", {}).get("status") == "verified"

        star_count = it.get("star_count") or it.get("stars") or 0
        pull_count = it.get("pull_count") or it.get("downloads") or 0

        # Some endpoints return strings like "10M+"; try to coerce
        if isinstance(pull_count, str):
            pc = pull_count.lower().replace("+", "").replace(",", "")
            if "k" in pc:
                pull_count = int(float(pc.replace("k", "")) * 1_000)
            elif "m" in pc:
                pull_count = int(float(pc.replace("m", "")) * 1_000_000)
            else:
                try:
                    pull_count = int(pc)
                except:
                    pull_count = 0

        updated_raw = it.get("last_updated") or it.get("updated_at") or it.get("modified_at")
        updated_dt = iso_to_dt(updated_raw)

        # Apply thresholds
        if star_count < MIN_STARS:
            continue
        if pull_count < MIN_PULLS:
            continue
        if not is_recent(updated_dt, MAX_LAST_UPDATED_DAYS):
            continue

        # For community: prefer verified publishers, but don't require it if we need volume
        filtered.append({
            "namespace": namespace,
            "name": name,
            "is_official": bool(is_official),
            "is_verified": bool(is_verified),
            "stars": star_count,
            "pulls": pull_count,
            "last_updated": updated_raw
        })
    return filtered

def get_official_candidates(limit=200):
    # library/* official images (Docker Hub "library")
    # Order by last_updated, larger page_size for breadth
    url = f"{API_REPOS}library/?page_size={limit}&ordering=last_updated"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return []
    items = r.json().get("results", [])
    # library namespace is official by definition
    for it in items:
        it["namespace"] = "library"
        it["is_official"] = True
    return filter_candidates(items, want_official=True)

def get_verified_community_candidates(limit=200):
    """
    Use the Hub 'products/search' API to find verified publishers with popular repos.
    Fallback to general repositories listing if needed.
    """
    # Try verified publishers with high stars/pulls
    params = {
        "type": "image",
        "page_size": limit,
        "facets": "publisher_status:verified",
        "sort": "updated_at:desc"
    }
    r = requests.get(API_SEARCH, params=params, timeout=30)
    items = []
    if r.status_code == 200:
        data = r.json()
        for p in data.get("summaries", []):
            # Try to map to a namespace/repo if present
            slug = p.get("slug")  # e.g., "bitnami/nginx"
            if not slug or "/" not in slug:
                continue
            ns, nm = slug.split("/", 1)
            items.append({
                "namespace": ns,
                "name": nm,
                "is_official": False,
                "is_verified": (p.get("publisher", {}).get("status") == "verified"),
                "star_count": p.get("stats", {}).get("stars", 0),
                "pull_count": p.get("stats", {}).get("pulls", 0),
                "last_updated": p.get("updated_at")
            })

    # If not enough, fallback to general repos listing (mixed quality)
    if len(items) < 50:
        url = f"{API_REPOS}?page_size={limit}&ordering=last_updated"
        r2 = requests.get(url, timeout=30)
        if r2.status_code == 200:
            items.extend(r2.json().get("results", []))

    return filter_candidates(items, want_official=False)

def choose_images(official_pool, community_pool, total_needed):
    want_official = int(total_needed * OFFICIAL_RATIO)
    want_community = total_needed - want_official

    random.shuffle(official_pool)
    random.shuffle(community_pool)

    chosen_official = official_pool[:want_official]
    chosen_community = community_pool[:want_community]

    chosen = chosen_official + chosen_community
    random.shuffle(chosen)
    return chosen

def pull_and_scan(namespace, repo_name, tag, out_dir):
    image_ref = f"{namespace}/{repo_name}:{tag}"
    safe_name = f"{namespace}_{repo_name}_{tag}".replace("/", "_").replace(":", "_")
    out_json = os.path.join(out_dir, f"{safe_name}.json")

 # Skip if this scan file already exists
    if os.path.exists(out_json):
        print(f"[~] Skipping {image_ref} (already scanned)")
        return False
   
    print(f"\n[+] Pulling {image_ref} ...")
    rc, out, err = run_cmd([DOCKER_CLI, "pull", image_ref])
    if rc != 0:
        print(f"[!] docker pull failed for {image_ref}: {err.strip()}")
        return False

    print(f"[+] Scanning {image_ref} with Trivy ...")
    rc, out, err = run_cmd([TRIVY_CLI, "image", "--quiet", "--format", "json", "-o", out_json, image_ref])
    if rc != 0:
        print(f"[!] trivy scan failed for {image_ref}: {err.strip()}")
        return False

    print(f"[✓] Saved scan: {out_json}")
    return True

def main():
    # Pre-flight checks
    if not check_cli_available(DOCKER_CLI):
        print("ERROR: docker CLI not found in PATH.")
        sys.exit(1)
    if not check_cli_available(TRIVY_CLI):
        print("ERROR: trivy CLI not found in PATH. Install via Homebrew.")
        sys.exit(1)

    ensure_dir(OUTPUT_DIR)

    print("[*] Gathering official candidates...")
    official_pool = get_official_candidates(limit=300)
    print(f"[i] Official pool after filters: {len(official_pool)}")

    print("[*] Gathering community candidates (verified preferred)...")
    community_pool = get_verified_community_candidates(limit=400)
    print(f"[i] Community pool after filters: {len(community_pool)}")

    if len(official_pool) + len(community_pool) < 20:
        print("ERROR: Too few candidates after filtering. Loosen thresholds or check network.")
        sys.exit(1)

    chosen = choose_images(official_pool, community_pool, TARGET_TOTAL_IMAGES)
    print(f"[i] Selected {len(chosen)} images (≈ {int(TARGET_TOTAL_IMAGES*OFFICIAL_RATIO)} official / {len(chosen) - int(TARGET_TOTAL_IMAGES*OFFICIAL_RATIO)} community)")

    # Resolve tags for each chosen image
    plan = []
    for item in chosen:
        ns = item["namespace"]
        nm = item["name"]
        ref, tag = pick_latest_tag(ns, nm)
        if ref and tag:
            plan.append((ns, nm, tag))
        # Pace API requests
        time.sleep(0.1)

    print(f"[i] Images with resolvable tags: {len(plan)}")

    # Pull + Scan (no running containers)
    successes, failures = 0, 0
    for (ns, nm, tag) in plan:
        ok = pull_and_scan(ns, nm, tag, OUTPUT_DIR)
        if ok:
            successes += 1
        else:
            failures += 1
        # Gentle pacing to avoid rate limits
        time.sleep(0.2)

    summary = {
        "total_planned": len(plan),
        "successes": successes,
        "failures": failures,
        "output_dir": os.path.abspath(OUTPUT_DIR),
        "filters": {
            "min_stars": MIN_STARS,
            "min_pulls": MIN_PULLS,
            "max_last_updated_days": MAX_LAST_UPDATED_DAYS
        }
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
