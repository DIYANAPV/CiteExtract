"""Generate a publication-quality pipeline diagram for CheckCitation.

Produces a clean, academic paper-style figure showing the full verification pipeline.
Run: python scripts/generate_pipeline_figure.py
Output: data/output/checkcitation_pipeline.png and .pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIG_WIDTH, FIG_HEIGHT = 16, 20
BG_COLOR = "#FFFFFF"

# Color palette (academic, muted)
C_INPUT = "#E8F0FE"       # light blue
C_INPUT_BD = "#4285F4"    # blue border
C_PARSE = "#FFF3E0"       # light orange
C_PARSE_BD = "#F4A236"    # orange border
C_QUICK = "#E8F5E9"       # light green
C_QUICK_BD = "#4CAF50"    # green border
C_STANDARD = "#E3F2FD"    # light blue
C_STANDARD_BD = "#2196F3" # blue border
C_AGENTIC = "#F3E5F5"     # light purple
C_AGENTIC_BD = "#9C27B0"  # purple border
C_TOOL = "#FAFAFA"        # near white
C_TOOL_BD = "#757575"     # gray border
C_OUTPUT = "#FFF8E1"      # light yellow
C_OUTPUT_BD = "#FF9800"   # orange border
C_VALID = "#4CAF50"
C_FABRICATED = "#F44336"
C_MISREP = "#FF9800"
C_UNVERIF = "#9E9E9E"
C_HEADER_TEXT = "#FFFFFF"
C_BODY_TEXT = "#333333"
C_ARROW = "#555555"


def draw_box(ax, x, y, w, h, header, body_lines, header_color, bg_color,
             border_color, header_fontsize=11, body_fontsize=9, bold_header=True):
    """Draw a box with a colored header bar and body text."""
    header_h = 0.45

    # Body box
    body = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.5,
    )
    ax.add_patch(body)

    # Header bar
    hdr = FancyBboxPatch(
        (x, y + h - header_h), w, header_h,
        boxstyle="round,pad=0.05",
        facecolor=header_color, edgecolor=header_color, linewidth=1.5,
    )
    ax.add_patch(hdr)

    # Overlay a thin strip to square off the bottom of the header
    strip = plt.Rectangle((x + 0.05, y + h - header_h), w - 0.1, 0.1,
                           facecolor=header_color, edgecolor="none", zorder=4)
    ax.add_patch(strip)

    # Header text
    weight = "bold" if bold_header else "normal"
    ax.text(x + w / 2, y + h - header_h / 2, header,
            ha="center", va="center", fontsize=header_fontsize,
            fontweight=weight, color=C_HEADER_TEXT, zorder=5)

    # Body text
    for i, line in enumerate(body_lines):
        ax.text(x + w / 2, y + h - header_h - 0.35 - i * 0.35, line,
                ha="center", va="center", fontsize=body_fontsize,
                color=C_BODY_TEXT, zorder=5)


def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW):
    """Draw a downward arrow between boxes."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>", color=color, lw=1.8,
            connectionstyle="arc3,rad=0",
        ),
        zorder=3,
    )


def draw_small_box(ax, x, y, w, h, text, bg, border, fontsize=8.5):
    """Draw a small labeled box (for tools, verdicts, etc.)."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04",
        facecolor=bg, edgecolor=border, linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=C_BODY_TEXT, zorder=5)


def draw_verdict_badge(ax, x, y, label, color, size=0.35):
    """Draw a small colored verdict badge."""
    circle = plt.Circle((x, y), size, facecolor=color, edgecolor="white",
                         linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, label[0], ha="center", va="center", fontsize=10,
            fontweight="bold", color="white", zorder=6)
    ax.text(x, y - size - 0.2, label, ha="center", va="center",
            fontsize=8, color=color, fontweight="bold", zorder=6)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # ===== TITLE =====
    ax.text(8, 19.5, "CheckCitation: Verification Pipeline",
            ha="center", va="center", fontsize=18, fontweight="bold",
            color="#333333")

    # ===== ROW 1: INPUT =====
    y_input = 17.5
    draw_box(ax, 1, y_input, 14, 1.5, "INPUT",
             ["PDF  ·  LaTeX (.tex)  ·  BibTeX (.bib)  ·  Plain Text (.txt)"],
             C_INPUT_BD, C_INPUT, C_INPUT_BD, header_fontsize=12, body_fontsize=11)

    # File type icons (text-based)
    icons = [("📄", "PDF", 3.5), ("📝", ".tex", 6.5), ("📋", ".bib", 9.5), ("📃", ".txt", 12.5)]
    for emoji, label, ix in icons:
        ax.text(ix, y_input + 0.45, label, ha="center", va="center",
                fontsize=10, color=C_INPUT_BD, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=C_INPUT_BD, linewidth=1))

    # Arrow: Input → Parse
    draw_arrow(ax, 8, y_input, 8, y_input - 0.3)

    # ===== ROW 2: PARSING (L1) =====
    y_parse = 14.8
    draw_box(ax, 1, y_parse, 14, 2.2, "L1: PARSING",
             [],
             C_PARSE_BD, C_PARSE, C_PARSE_BD, header_fontsize=12)

    # Sub-boxes inside parsing
    draw_small_box(ax, 1.5, y_parse + 0.15, 3.2, 0.7,
                   "GROBID\n(PDF → XML)", "#FFF8E1", C_PARSE_BD, fontsize=9)
    draw_small_box(ax, 5.0, y_parse + 0.15, 3.2, 0.7,
                   "LLM Reparse\n(clean fields)", "#FFF8E1", C_PARSE_BD, fontsize=9)
    draw_small_box(ax, 8.5, y_parse + 0.15, 3.5, 0.7,
                   "Format Detection\n(APA/Vancouver/IEEE...)", "#FFF8E1", C_PARSE_BD, fontsize=9)
    draw_small_box(ax, 12.3, y_parse + 0.15, 2.4, 0.7,
                   "Citation\nDetection", "#FFF8E1", C_PARSE_BD, fontsize=9)

    # Output label
    ax.text(8, y_parse + 1.15, "References + Citing Contexts + Citation Format",
            ha="center", va="center", fontsize=9.5, color=C_PARSE_BD,
            fontstyle="italic")

    # Arrow: Parse → Modes
    draw_arrow(ax, 8, y_parse, 8, y_parse - 0.3)

    # ===== ROW 3: THREE MODES =====
    y_mode = 11.8
    mode_w = 4.0
    mode_h = 2.5

    # Quick mode
    draw_box(ax, 0.8, y_mode, mode_w, mode_h, "QUICK MODE",
             ["Free — no LLM needed", "",
              "L2: Existence Check",
              "  CrossRef → S2 → OpenAlex → PubMed",
              "L3: Metadata Validation",
              "  Field-by-field comparison"],
             C_QUICK_BD, C_QUICK, C_QUICK_BD)

    # Standard mode
    draw_box(ax, 5.5, y_mode, mode_w + 1, mode_h, "STANDARD MODE",
             ["~$0.01–0.05/paper", "",
              "Quick mode checks PLUS:",
              "  Full-text retrieval (S2/Unpaywall/arXiv)",
              "  BM25 + Dense hybrid retrieval",
              "  LLM claim verification"],
             C_STANDARD_BD, C_STANDARD, C_STANDARD_BD)

    # Agentic mode
    draw_box(ax, 11.2, y_mode, mode_w, mode_h, "AGENTIC MODE",
             ["~$0.17/paper", "",
              "LLM agent per reference",
              "  with database + comparison",
              "  + retrieval tools",
              "  Investigation-based"],
             C_AGENTIC_BD, C_AGENTIC, C_AGENTIC_BD)

    # Arrows from parse to each mode
    draw_arrow(ax, 5, y_parse, 2.8, y_mode + mode_h + 0.05, C_QUICK_BD)
    draw_arrow(ax, 8, y_parse, 8, y_mode + mode_h + 0.05, C_STANDARD_BD)
    draw_arrow(ax, 11, y_parse, 13.2, y_mode + mode_h + 0.05, C_AGENTIC_BD)

    # ===== ROW 4: AGENTIC DETAIL =====
    y_agent = 8.0
    draw_box(ax, 0.8, y_agent, 14.4, 3.3, "AGENTIC AGENT LOOP  (max 8 rounds per reference)",
             [],
             C_AGENTIC_BD, "#F9F0FA", C_AGENTIC_BD, header_fontsize=11)

    # Database tools
    ax.text(3.5, y_agent + 2.35, "Database Tools", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_AGENTIC_BD)
    db_tools = ["CrossRef", "Semantic\nScholar", "OpenAlex", "PubMed"]
    for i, t in enumerate(db_tools):
        draw_small_box(ax, 1.3 + i * 1.8, y_agent + 1.2, 1.5, 0.8,
                       t, "#EDE7F6", C_AGENTIC_BD, fontsize=8)

    # Comparison tools
    ax.text(9.2, y_agent + 2.35, "Comparison Tools", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_AGENTIC_BD)
    draw_small_box(ax, 8.0, y_agent + 1.2, 2.8, 0.8,
                   "compare_titles\n(word-level diff)", "#EDE7F6", C_AGENTIC_BD, fontsize=8)
    draw_small_box(ax, 8.0, y_agent + 0.2, 2.8, 0.8,
                   "compare_authors\n(format-aware canonical)", "#EDE7F6", C_AGENTIC_BD, fontsize=8)

    # Other tools
    ax.text(13, y_agent + 2.35, "Other Tools", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_AGENTIC_BD)
    draw_small_box(ax, 11.5, y_agent + 1.2, 2.8, 0.8,
                   "verify_web_source\n(URL + Wayback)", "#EDE7F6", C_AGENTIC_BD, fontsize=8)
    draw_small_box(ax, 11.5, y_agent + 0.2, 2.8, 0.8,
                   "retrieve_passages\n(BM25 + dense + RRF)", "#EDE7F6", C_AGENTIC_BD, fontsize=8)

    # Agent flow text
    ax.text(3.5, y_agent + 0.5, "LLM decides →\ntool call →\nresult → repeat",
            ha="center", va="center", fontsize=9, color="#666666",
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#E0E0E0", linewidth=1))

    # Arrows: all modes → classification
    y_class = 5.5
    draw_arrow(ax, 2.8, y_mode, 6, y_class + 2.0, C_QUICK_BD)
    draw_arrow(ax, 8, y_mode, 8, y_class + 2.0, C_STANDARD_BD)
    draw_arrow(ax, 8, y_agent, 10, y_class + 2.0, C_AGENTIC_BD)

    # ===== ROW 5: CLASSIFICATION =====
    draw_box(ax, 2.5, y_class, 11, 1.5, "L5: CLASSIFICATION",
             ["Decision tree combines all signals → verdict per reference"],
             "#455A64", "#ECEFF1", "#455A64", header_fontsize=12)

    # Arrow: classification → report
    draw_arrow(ax, 8, y_class, 8, y_class - 0.3)

    # ===== ROW 6: OUTPUT =====
    y_out = 2.5
    draw_box(ax, 1, y_out, 14, 2.5, "OUTPUT: VERIFICATION REPORT",
             [],
             C_OUTPUT_BD, C_OUTPUT, C_OUTPUT_BD, header_fontsize=12)

    # Verdict badges
    verdicts = [
        ("VALID", C_VALID, "✓", 3.5),
        ("FABRICATED", C_FABRICATED, "✗", 6.5),
        ("MISREPRESENTED", C_MISREP, "!", 9.5),
        ("UNVERIFIABLE", C_UNVERIF, "?", 12.5),
    ]
    for label, color, symbol, vx in verdicts:
        draw_small_box(ax, vx - 1.1, y_out + 0.3, 2.2, 1.2,
                       f"{symbol}\n{label}", color + "22", color, fontsize=9)
        ax.text(vx, y_out + 1.1, symbol, ha="center", va="center",
                fontsize=16, fontweight="bold", color=color, zorder=6)
        ax.text(vx, y_out + 0.55, label, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=color, zorder=6)

    # Bottom text
    ax.text(8, y_out - 0.15,
            "integrity_score  ·  risk_level  ·  field_discrepancies  ·  evidence trail",
            ha="center", va="center", fontsize=9, color="#888888", fontstyle="italic")

    # ===== SAVE =====
    import os
    os.makedirs("data/output", exist_ok=True)

    fig.savefig("data/output/checkcitation_pipeline.png",
                dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    fig.savefig("data/output/checkcitation_pipeline.pdf",
                bbox_inches="tight", facecolor=BG_COLOR)
    print("Saved: data/output/checkcitation_pipeline.png")
    print("Saved: data/output/checkcitation_pipeline.pdf")
    plt.close()


if __name__ == "__main__":
    main()
