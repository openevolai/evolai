"""Metadata compression utilities for Bittensor commitments

Bittensor commitments have a 128-byte limit for Raw data type.
These functions compress/decompress model metadata to fit within this limit.

NOTE: Each miner can only register for ONE track (transformer OR mamba2).
      Metadata will always be single-track only (~85-100 bytes).
"""

import json
from datetime import datetime
from typing import Dict, Union


def compress_metadata(metadata: dict) -> str:
    """
    Compress metadata to fit within 128 byte Bittensor commitment limit.
    
    Stores full model_name and full revision. Validators will verify 
    HuggingFace upload times themselves.
    
    Uses:
    - Short field names (t/m2 for tracks, m/r for model fields)
    - Full 40-char commit hashes preserved
    - No version field (implied v1.0)
    - No whitespace in JSON
    
    Args:
        metadata: Full metadata dict with ONE track ("transformer" or "mamba2")
    
    Returns:
        Compressed JSON string (~85-100 bytes for single track)
        
    Example:
        Input: {
            "transformer": {
                "model_name": "user/evolai-model",
                "revision": "4141d3484d6f6b8949c981e70b029dc30e3697bf"
            }
        }
        Output: {"t":{"m":"user/evolai-model","r":"4141d3484d6f6b8949c981e70b029dc30e3697bf"}}
        (~90 bytes for single track, ~180 bytes for both tracks - need short names for both)
    """
    compressed = {}
    
    # Handle transformer track
    if "transformer" in metadata:
        t = metadata["transformer"]
        compressed["t"] = {
            "m": t["model_name"],
            "r": t.get("revision") or "main"
        }
    
    # Handle Mamba2 track
    if "mamba2" in metadata:
        s = metadata["mamba2"]
        compressed["m2"] = {
            "m": s["model_name"],
            "r": s.get("revision") or "main"
        }
    
    return json.dumps(compressed, separators=(',', ':'))  # No whitespace


def decompress_metadata(compressed_str: Union[str, bytes]) -> dict:
    """
    Decompress metadata from chain format back to full structure.

    Accepts either a JSON string or raw UTF-8 bytes (as returned by the
    Bittensor RawN commitment field).

    Args:
        compressed_str: Compressed JSON string or bytes from Bittensor commitment
    
    Returns:
        Full metadata dict with "transformer" and/or "mamba2" tracks
        
    Example:
        Input: {"t":{"m":"user/model","r":"abc123def456"}}
        Output: {
            "transformer": {
                "model_name": "user/model",
                "revision": "abc123def456"
            },
            "version": "1.0"
        }
    """
    compressed = json.loads(compressed_str)
    metadata = {}
    
    # Expand transformer track
    if "t" in compressed:
        t = compressed["t"]
        metadata["transformer"] = {
            "model_name": t["m"],
            "revision": t["r"]
        }
    
    # Expand Mamba2 track
    if "m2" in compressed:
        s = compressed["m2"]
        metadata["mamba2"] = {
            "model_name": s["m"],
            "revision": s["r"]
        }
    
    # Legacy support: expand old "s" (srrm) to mamba2
    if "s" in compressed:
        s = compressed["s"]
        metadata["mamba2"] = {
            "model_name": s["m"],
            "revision": s["r"]
        }
    
    metadata["version"] = compressed.get("v", "1.0")
    
    return metadata
