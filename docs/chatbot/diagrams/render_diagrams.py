"""
Generate PNG images from Mermaid source files using the mermaid.ink rendering API.
"""

import base64
import os
import requests
import zlib

DIAGRAMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Docs", "chatbot", "diagrams")

DIAGRAMS = [
    "architecture.mmd",
    "sequence.mmd",
    "component.mmd",
    "deployment.mmd",
]

def encode_mermaid(source: str) -> str:
    """Encode Mermaid source for mermaid.ink URL using the pako zlib method."""
    # mermaid.ink uses: base64(pako_deflate(source))
    compressed = zlib.compress(source.encode("utf-8"), level=9)
    # Remove zlib header (first 2 bytes) and checksum (last 4 bytes) to get raw deflate
    # Actually, mermaid.ink expects the full zlib compressed data base64-encoded
    encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")
    return encoded

def render_mermaid_to_png(source: str, output_path: str) -> bool:
    """Render a Mermaid diagram to PNG using mermaid.ink API."""
    encoded = encode_mermaid(source)
    url = f"https://mermaid.ink/img/{encoded}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"  SUCCESS: Saved {output_path} ({len(response.content)} bytes)")
            return True
        else:
            print(f"  FAILED: HTTP {response.status_code} for {output_path}")
            # Try alternative encoding method (plain base64 without compression)
            plain_encoded = base64.urlsafe_b64encode(source.encode("utf-8")).decode("utf-8")
            url2 = f"https://mermaid.ink/img/base64:{plain_encoded}"
            response2 = requests.get(url2, timeout=30)
            if response2.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response2.content)
                print(f"  SUCCESS (alt encoding): Saved {output_path} ({len(response2.content)} bytes)")
                return True
            else:
                print(f"  FAILED (alt encoding): HTTP {response2.status_code}")
                return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    print(f"Diagrams directory: {DIAGRAMS_DIR}")
    print(f"Generating PNG images from Mermaid source files...\n")
    
    results = {}
    for diagram_file in DIAGRAMS:
        mmd_path = os.path.join(DIAGRAMS_DIR, diagram_file)
        png_filename = diagram_file.replace(".mmd", ".png")
        png_path = os.path.join(DIAGRAMS_DIR, png_filename)
        
        if not os.path.exists(mmd_path):
            print(f"  SKIP: {mmd_path} not found")
            results[diagram_file] = False
            continue
        
        with open(mmd_path, "r", encoding="utf-8") as f:
            source = f.read()
        
        print(f"Processing: {diagram_file} -> {png_filename}")
        success = render_mermaid_to_png(source, png_path)
        results[diagram_file] = success
    
    print(f"\n--- Results ---")
    for diagram_file, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {diagram_file}: {status}")
    
    all_success = all(results.values())
    if all_success:
        print(f"\nAll PNG images generated successfully!")
    else:
        print(f"\nSome PNG images failed to generate.")
        print(f"Alternative: Paste .mmd content into https://mermaid.live to export manually.")
    
    return all_success

if __name__ == "__main__":
    main()