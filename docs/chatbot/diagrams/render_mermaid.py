"""
Generate PNG images from Mermaid (.mmd) source files using the Kroki rendering service.

Kroki supports Mermaid diagrams and its POST API:
https://kroki.io/mermaid/png

This is more reliable than mermaid.ink for complex diagrams.
"""

import base64
import os
import sys
import requests
import json

DIAGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))

MMD_FILES = [
    "architecture_render.mmd",
    "sequence_render.mmd",
    "component_render.mmd",
    "deployment_render.mmd",
]

PNG_OUTPUTS = [
    "architecture.png",
    "sequence.png",
    "component.png",
    "deployment.png",
]

KROKI_URL = "https://kroki.io/mermaid/png"


def render_mermaid_to_png(mmd_content: str, output_path: str) -> bool:
    """Render a Mermaid diagram to PNG using the Kroki API.
    
    Kroki accepts POST requests with JSON body containing the diagram source.
    
    Args:
        mmd_content: The Mermaid source code
        output_path: Path to save the PNG file
        
    Returns:
        True if successful, False otherwise
    """
    payload = {
        "diagram_source": mmd_content,
        "diagram_type": "mermaid",
        "output_format": "png"
    }
    
    try:
        print(f"  Sending POST request to Kroki API...")
        response = requests.post(
            KROKI_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            file_size = os.path.getsize(output_path)
            print(f"  [OK] Saved: {output_path} ({file_size} bytes)")
            return True
        elif response.status_code == 400:
            print(f"  [FAIL] HTTP 400 - Bad Request")
            error_detail = response.text[:500]
            print(f"    Error detail: {error_detail}")
            # Try alternative: GET request with base64 encoding
            print(f"  Trying alternative GET approach...")
            encoded = base64.urlsafe_b64encode(mmd_content.encode("utf-8")).decode("ascii")
            alt_url = f"https://kroki.io/mermaid/png/{encoded}"
            alt_response = requests.get(alt_url, timeout=60)
            if alt_response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(alt_response.content)
                file_size = os.path.getsize(output_path)
                print(f"  [OK] Saved via GET: {output_path} ({file_size} bytes)")
                return True
            else:
                print(f"  [FAIL] Alternative GET also failed: HTTP {alt_response.status_code}")
                return False
        else:
            print(f"  [FAIL] HTTP {response.status_code}")
            print(f"    Response: {response.text[:300]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  [FAIL] Network error: {e}")
        return False


def main():
    print("=" * 60)
    print("Mermaid Diagram PNG Generator (using Kroki API)")
    print("=" * 60)
    print(f"Source directory: {DIAGRAMS_DIR}")
    print()
    
    success_count = 0
    fail_count = 0
    
    for i, mmd_file in enumerate(MMD_FILES):
        png_file = PNG_OUTPUTS[i]
        mmd_path = os.path.join(DIAGRAMS_DIR, mmd_file)
        png_path = os.path.join(DIAGRAMS_DIR, png_file)
        
        print(f"Processing: {mmd_file} -> {png_file}")
        
        if not os.path.exists(mmd_path):
            print(f"  [FAIL] Source file not found: {mmd_path}")
            fail_count += 1
            continue
        
        with open(mmd_path, "r", encoding="utf-8") as f:
            mmd_content = f.read()
        
        if render_mermaid_to_png(mmd_content, png_path):
            success_count += 1
        else:
            fail_count += 1
        
        print()
    
    print("=" * 60)
    print(f"Results: {success_count} succeeded, {fail_count} failed")
    print("=" * 60)
    
    # List all PNG files in the directory
    print("\nGenerated PNG files:")
    png_found = 0
    for f in sorted(os.listdir(DIAGRAMS_DIR)):
        if f.endswith(".png"):
            full_path = os.path.join(DIAGRAMS_DIR, f)
            size = os.path.getsize(full_path)
            print(f"  {f} ({size} bytes)")
            print(f"    Full path: {full_path}")
            png_found += 1
    
    if png_found == 0:
        print("  No PNG files found.")
    
    return fail_count == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)