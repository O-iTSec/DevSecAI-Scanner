import os
import json
import pandas as pd

# -----------------------------------
# CONFIG
# -----------------------------------
# Automatically resolve to project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCAN_DIRS = [
    os.path.join(BASE_DIR, "data/scans/data/scans"),
    os.path.join(BASE_DIR, "data/scans"),
    os.path.join(BASE_DIR, "data/scans_batch2"),
    os.path.join(BASE_DIR, "data/scans_batch3"),
]
OUTPUT_CSV = os.path.join(BASE_DIR, "data/dataset.csv")

# -----------------------------------
# Helper Functions
# -----------------------------------
def extract_image_features(scan_json):
    """Extract aggregated vulnerability features from a Trivy JSON scan."""
    image_name = scan_json.get("ArtifactName", "unknown")
    results = scan_json.get("Results", [])
    
    # Initialize counters
    sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    total_vulns, fixable_vulns = 0, 0
    base_os, distro = None, None

    for res in results:
        # Capture OS/Distro info if available
        if not base_os and res.get("Type") == "os-pkgs":
            base_os = res.get("Target")
            distro = res.get("Class")

        vulns = res.get("Vulnerabilities") or []
        total_vulns += len(vulns)
        for v in vulns:
            sev = v.get("Severity", "UNKNOWN").upper()
            if sev not in sev_counts:
                sev = "UNKNOWN"
            sev_counts[sev] += 1
            if v.get("FixedVersion"):
                fixable_vulns += 1

    # Derived metrics
    fixable_ratio = round(fixable_vulns / total_vulns, 3) if total_vulns > 0 else 0
    vuln_density = total_vulns  # total per image
    high_risk_ratio = (sev_counts["CRITICAL"] + sev_counts["HIGH"]) / total_vulns if total_vulns > 0 else 0

    return {
        "image_name": image_name,
        "base_os": base_os,
        "distro": distro,
        "total_vulns": total_vulns,
        "critical": sev_counts["CRITICAL"],
        "high": sev_counts["HIGH"],
        "medium": sev_counts["MEDIUM"],
        "low": sev_counts["LOW"],
        "unknown": sev_counts["UNKNOWN"],
        "fixable_vulns": fixable_vulns,
        "fixable_ratio": fixable_ratio,
        "high_risk_ratio": round(high_risk_ratio, 3),
    }

def load_scans(scan_dirs):
    """Load and parse all JSON scans across multiple directories."""
    data = []
    seen = set()

    for folder in scan_dirs:
        if not os.path.exists(folder):
            continue
        json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
        if not json_files:
            continue
        for file in json_files:
            file_path = os.path.join(folder, file)
            try:
                with open(file_path, "r") as f:
                    scan_json = json.load(f)
                image_name = scan_json.get("ArtifactName", file)
                if image_name in seen:
                    continue
                seen.add(image_name)
                row = extract_image_features(scan_json)
                data.append(row)
                print(f"[+] Parsed: {file} — {row['total_vulns']} vulns")
            except Exception as e:
                print(f"[!] Failed to parse {file}: {e}")
                continue

    return data

# -----------------------------------
# MAIN
# -----------------------------------
if __name__ == "__main__":
    records = load_scans(SCAN_DIRS)

    if not records:
        print("No valid scan files found. Check paths.")
        exit(1)

    df = pd.DataFrame(records)
    df.drop_duplicates(subset="image_name", inplace=True)

    # Derived additional features
    df["severity_index"] = (
        4 * df["critical"] + 3 * df["high"] + 2 * df["medium"] + df["low"]
    )
    df["risk_score"] = (
        df["high_risk_ratio"] * 0.7 + df["fixable_ratio"] * 0.3
    ).round(3)

    df.sort_values(by="risk_score", ascending=False, inplace=True)

    # Save output
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[✓] Dataset created: {OUTPUT_CSV} ({len(df)} images)")
