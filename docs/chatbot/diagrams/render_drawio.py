"""
Render Draw.io XML files to PNG using Kroki API.
Kroki supports 'diagramsnet' format (Draw.io XML).
POST https://kroki.io/diagramsnet/png with JSON payload.
"""

import os
import sys
import json
import base64
import requests

DIAGRAMS_DIR = os.path.dirname(os.path.abspath(__file__))
KROKI_URL = "https://kroki.io/diagramsnet/png"

DRAWIO_FILES = [
    "architecture.drawio",
    "sequence.drawio",
    "component.drawio",
    "deployment.drawio",
    "dataflow.drawio",
]

def render_drawio_to_png(drawio_path, png_path):
    """Render a Draw.io XML file to PNG via Kroki API."""
    with open(drawio_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    # Kroki expects the diagram source as a base64-encoded string
    encoded = base64.b64encode(xml_content.encode("utf-8")).decode("utf-8")

    payload = {
        "diagram_source": encoded,
        "diagram_type": "diagramsnet",
        "output_format": "png"
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(KROKI_URL, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            with open(png_path, "wb") as f:
                f.write(response.content)
            size = os.path.getsize(png_path)
            print(f"[OK] {os.path.basename(png_path)} ({size:,} bytes)")
            return True
        else:
            print(f"[FAIL] {os.path.basename(drawio_path)} - HTTP {response.status_code}")
            print(f"       Response: {response.text[:200]}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"[FAIL] {os.path.basename(drawio_path)} - {str(e)[:100]}")
        return False


def main():
    print("=" * 60)
    print("Rendering Draw.io XML files to PNG via Kroki API")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for drawio_file in DRAWIO_FILES:
        drawio_path = os.path.join(DIAGRAMS_DIR, drawio_file)
        png_filename = drawio_file.replace(".drawio", "_drawio.png")
        png_path = os.path.join(DIAGRAMS_DIR, png_filename)

        if not os.path.exists(drawio_path):
            print(f"[SKIP] {drawio_file} - file not found")
            fail_count += 1
            continue

        print(f"\nRendering: {drawio_file} -> {png_filename}")
        if render_drawio_to_png(drawio_path, png_path):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"Results: {success_count} succeeded, {fail_count} failed")
    print("=" * 60)

    # List all PNG files in the directory
    print("\nAll PNG files in diagrams directory:")
    for f in sorted(os.listdir(DIAGRAMS_DIR)):
        if f.endswith(".png"):
            size = os.path.getsize(os.path.join(DIAGRAMS_DIR, f))
            print(f"  {f} ({size:,} bytes)")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())