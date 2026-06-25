from __future__ import annotations

from pathlib import Path
import os

from .chart_specs import ChartSpec


def render_chart(spec: ChartSpec, rows: list[tuple], output_dir: Path, *, stem: str = "chart") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row[0]) for row in rows]
    values = [float(row[1] or 0) for row in rows]
    png_path = output_dir / f"{stem}.png"

    try:
        mpl_config = output_dir / "matplotlib_config"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        fallback = output_dir / f"{stem}.txt"
        fallback.write_text(_render_text_fallback(spec, labels, values), encoding="utf-8")
        return fallback

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if spec.chart_type == "pie":
        ax.pie(values, labels=labels, autopct="%1.1f%%")
        ax.set_title(spec.title)
    elif spec.chart_type == "line":
        ax.plot(labels, values, marker="o")
        ax.set_title(spec.title)
        ax.set_xlabel(spec.x_label or spec.x_field)
        ax.set_ylabel(spec.y_label or spec.aggregation)
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.bar(labels, values)
        ax.set_title(spec.title)
        ax.set_xlabel(spec.x_label or spec.x_field)
        ax.set_ylabel(spec.y_label or spec.aggregation)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(png_path, format="png", bbox_inches="tight")
    plt.close(fig)
    return png_path


def _render_text_fallback(spec: ChartSpec, labels: list[str], values: list[float]) -> str:
    lines = [spec.title, f"type={spec.chart_type}"]
    lines.extend(f"{label}: {value}" for label, value in zip(labels, values))
    return "\n".join(lines) + "\n"
