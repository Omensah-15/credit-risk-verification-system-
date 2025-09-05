import hashlib
import json
from typing import Dict, Any

FIELDS_TO_REMOVE = ["submission_timestamp", "applicant_id", "customer_id"]

def generate_data_hash(data: Dict[str, Any]) -> str:
    """
    Deterministic SHA-256 hash of applicant data (excludes transient fields).
    """
    data_copy = dict(data)
    for f in FIELDS_TO_REMOVE:
        data_copy.pop(f, None)
    canonical = json.dumps(data_copy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_data_hash(data: Dict[str, Any], original_hash: str) -> bool:
    return generate_data_hash(data) == original_hash
