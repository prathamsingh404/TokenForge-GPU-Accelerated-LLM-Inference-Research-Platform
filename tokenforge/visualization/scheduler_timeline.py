"""
Scheduler timeline visualization.

Renders a Gantt-chart style timeline showing:
- Request arrivals (vertical markers)
- Prefill phase per request
- Decode phase per request
- Completion times
- Queue wait time (gap between arrival and first processing)

Produces an interactive HTML timeline or static matplotlib plot.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class TimelineEvent:
    """A single event in the scheduler timeline."""
    request_id: str
    event_type: str  # arrival, prefill_start, decode_start, completion
    timestamp: float
    priority: int = 0


class SchedulerTimeline:
    """
    Visual timeline of scheduler behavior.

    Shows how requests flow through the system, enabling visual
    comparison of different scheduling strategies.
    """

    def __init__(self):
        self._events: list[TimelineEvent] = []

    def add_event(self, request_id: str, event_type: str,
                  timestamp: float, priority: int = 0):
        self._events.append(TimelineEvent(
            request_id=request_id,
            event_type=event_type,
            timestamp=timestamp,
            priority=priority,
        ))

    def render_matplotlib(
        self,
        output_path: Optional[str] = None,
        max_requests: int = 50,
        figsize: tuple[int, int] = (16, 8),
    ):
        """Render timeline as a matplotlib Gantt chart."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import numpy as np
        except ImportError:
            print("matplotlib required. Install with: pip install matplotlib")
            return

        # Group events by request
        requests = {}
        for e in self._events:
            if e.request_id not in requests:
                requests[e.request_id] = {}
            requests[e.request_id][e.event_type] = e.timestamp

        # Limit display
        request_ids = list(requests.keys())[:max_requests]

        fig, ax = plt.subplots(figsize=figsize)

        colors = {
            "queue": "#4a5568",      # Gray — waiting
            "prefill": "#6366f1",    # Indigo — prefill
            "decode": "#10b981",     # Emerald — decode
        }

        for i, rid in enumerate(request_ids):
            r = requests[rid]
            y = i

            arrival = r.get("arrival", 0)
            prefill_start = r.get("prefill_start", arrival)
            decode_start = r.get("decode_start", prefill_start)
            completion = r.get("completion", decode_start)

            # Queue wait
            if prefill_start > arrival:
                ax.barh(y, prefill_start - arrival, left=arrival,
                       color=colors["queue"], height=0.6, alpha=0.5)

            # Prefill
            if decode_start > prefill_start:
                ax.barh(y, decode_start - prefill_start, left=prefill_start,
                       color=colors["prefill"], height=0.6)

            # Decode
            if completion > decode_start:
                ax.barh(y, completion - decode_start, left=decode_start,
                       color=colors["decode"], height=0.6)

        ax.set_yticks(range(len(request_ids)))
        ax.set_yticklabels(request_ids, fontsize=7)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_title("Scheduler Timeline", fontsize=14, fontweight="bold")
        ax.invert_yaxis()

        # Legend
        legend_items = [
            mpatches.Patch(color=colors["queue"], label="Queue Wait", alpha=0.5),
            mpatches.Patch(color=colors["prefill"], label="Prefill"),
            mpatches.Patch(color=colors["decode"], label="Decode"),
        ]
        ax.legend(handles=legend_items, loc="upper right")

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Timeline saved to {output_path}")

        plt.close(fig)
        return fig

    def render_html(self, output_path: Optional[str] = None) -> str:
        """Render timeline as standalone interactive HTML."""
        requests = {}
        for e in self._events:
            if e.request_id not in requests:
                requests[e.request_id] = {"priority": 0}
            requests[e.request_id][e.event_type] = e.timestamp
            if e.event_type == "arrival":
                requests[e.request_id]["priority"] = e.priority

        rows_html = []
        max_time = max(e.timestamp for e in self._events) if self._events else 1.0

        for rid, r in list(requests.items())[:100]:
            arrival = r.get("arrival", 0)
            prefill = r.get("prefill_start", arrival)
            decode = r.get("decode_start", prefill)
            completion = r.get("completion", decode)

            def pct(t):
                return f"{(t / max_time) * 100:.2f}%"

            queue_w = pct(prefill - arrival)
            queue_l = pct(arrival)
            prefill_w = pct(decode - prefill)
            prefill_l = pct(prefill)
            decode_w = pct(completion - decode)
            decode_l = pct(decode)

            rows_html.append(f"""
            <div class="timeline-row">
                <span class="req-label">{rid}</span>
                <div class="bar-container">
                    <div class="bar queue" style="left:{queue_l};width:{queue_w}"></div>
                    <div class="bar prefill" style="left:{prefill_l};width:{prefill_w}"></div>
                    <div class="bar decode" style="left:{decode_l};width:{decode_w}"></div>
                </div>
            </div>""")

        html = f"""<!DOCTYPE html>
<html><head><style>
body {{ font-family: 'Inter', sans-serif; background: #0f172a; color: white; padding: 20px; }}
h1 {{ color: #10b981; }}
.timeline-row {{ display: flex; align-items: center; margin: 2px 0; height: 20px; }}
.req-label {{ width: 100px; font-size: 11px; color: #94a3b8; font-family: monospace; }}
.bar-container {{ flex: 1; position: relative; height: 16px; background: #1e293b; border-radius: 3px; }}
.bar {{ position: absolute; height: 100%; border-radius: 3px; }}
.queue {{ background: #4a5568; opacity: 0.5; }}
.prefill {{ background: #6366f1; }}
.decode {{ background: #10b981; }}
.legend {{ display: flex; gap: 20px; margin: 20px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 13px; }}
.legend-box {{ width: 16px; height: 16px; border-radius: 3px; }}
</style></head><body>
<h1>Scheduler Timeline</h1>
<div class="legend">
    <div class="legend-item"><div class="legend-box" style="background:#4a5568"></div>Queue</div>
    <div class="legend-item"><div class="legend-box" style="background:#6366f1"></div>Prefill</div>
    <div class="legend-item"><div class="legend-box" style="background:#10b981"></div>Decode</div>
</div>
{''.join(rows_html)}
</body></html>"""

        if output_path:
            Path(output_path).write_text(html)
            print(f"Timeline HTML saved to {output_path}")

        return html
