"""
Interactive HTML profiling reports using PyTorch Profiler traces.
Generates an HTML report that includes kernel-level execution times.
"""

import json
from pathlib import Path
from typing import Optional

def generate_html_report(trace_json_path: str, output_html_path: str, title: str = "TokenForge Profiling Report"):
    """
    Parses a Chrome Trace Event format JSON (from PyTorch Profiler)
    and generates a standalone HTML report.
    """
    trace_path = Path(trace_json_path)
    out_path = Path(output_html_path)

    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_json_path}")

    with open(trace_path, 'r') as f:
        trace_data = json.load(f)
        
    # The trace JSON usually has 'traceEvents' list.
    events = trace_data.get('traceEvents', trace_data)
    if not isinstance(events, list):
        events = []

    # Filter out CUDA kernels and CPU ops
    kernels = [e for e in events if e.get('cat') == 'Kernel']
    
    # Sort kernels by duration
    for k in kernels:
        k['dur'] = k.get('dur', 0)
    kernels.sort(key=lambda x: x['dur'], reverse=True)
    
    top_kernels = kernels[:50]  # Top 50 longest kernels

    # Build HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 2rem; background: #0f172a; color: #e2e8f0; }}
            h1 {{ color: #10b981; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; background: #1e293b; }}
            th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #334155; }}
            th {{ background-color: #0f172a; color: #38bdf8; }}
            .dur-bar {{ background-color: #ef4444; height: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>Total Trace Events: {len(events)} | Kernels: {len(kernels)}</p>
        
        <h2>Top 50 Longest CUDA Kernels</h2>
        <table>
            <thead>
                <tr>
                    <th>Kernel Name</th>
                    <th>Duration (us)</th>
                    <th>Visual</th>
                </tr>
            </thead>
            <tbody>
    """
    
    max_dur = top_kernels[0]['dur'] if top_kernels else 1
    
    for k in top_kernels:
        name = k.get('name', 'Unknown')
        # Shorten mangled names
        if len(name) > 80:
            name = name[:77] + "..."
        dur = k['dur']
        bar_width = max(1, int((dur / max_dur) * 100))
        html_content += f"""
                <tr>
                    <td>{name}</td>
                    <td>{dur}</td>
                    <td><div class="dur-bar" style="width: {bar_width}%;"></div></td>
                </tr>
        """
        
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(html_content)

    return str(out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate interactive profiling report")
    parser.add_argument("--trace", type=str, required=True, help="Path to PyTorch Profiler trace JSON")
    parser.add_argument("--out", type=str, required=True, help="Path to output HTML")
    args = parser.parse_args()
    
    generate_html_report(args.trace, args.out)
    print(f"Report generated at {args.out}")
