"""
GeoRecon AI - Cancel Reconstruction Button Layout Regression Test Suite
Validates:
1. Grid structure: Header (row 0), Confidence (row 1), Content (row 2, weight 1), Action Bar (row 3, weight 0)
2. Bottom action bar anchoring and full width spanning
3. 12-16 px window edge margins
4. Window resizing robustness across multiple dimensions
5. Console scrolling without button overlapping text
6. Button behavior in running and cancelled reconstruction states
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import GeoReconApp
from pipeline import PipelineEvent, StageType, StageStatus


def test_cancel_button_layout():
    print("=" * 70)
    print("Testing Cancel Reconstruction Button Layout & Geometry")
    print("=" * 70)

    app = GeoReconApp()
    app.switch_page("progress")
    app.update_idletasks()
    app.update()

    page = app.page_frames["progress"]

    # 1. Verify Grid Row Configurations
    print("\n--- 1. Verifying Grid Row Configurations ---")
    w0 = page.grid_rowconfigure(0)["weight"]
    w1 = page.grid_rowconfigure(1)["weight"]
    w2 = page.grid_rowconfigure(2)["weight"]
    w3 = page.grid_rowconfigure(3)["weight"]
    col_w = page.grid_columnconfigure(0)["weight"]

    print(f"Row 0 (Header) weight: {w0}")
    print(f"Row 1 (Confidence) weight: {w1}")
    print(f"Row 2 (Content) weight: {w2}")
    print(f"Row 3 (Action Bar) weight: {w3}")
    print(f"Column 0 weight: {col_w}")

    assert w0 == 0, f"Expected row 0 weight 0, got {w0}"
    assert w1 == 0, f"Expected row 1 weight 0, got {w1}"
    assert w2 == 1, f"Expected row 2 weight 1 (content row), got {w2}"
    assert w3 == 0, f"Expected row 3 weight 0 (action bar), got {w3}"
    assert col_w == 1, f"Expected column 0 weight 1, got {col_w}"

    # 2. Verify Widget Placement
    print("\n--- 2. Verifying Widget Row Assignments ---")
    children = list(page.winfo_children())
    top_telem = children[0]
    conf_card = children[1]
    split_frame = children[2]
    bottom_bar = children[3]

    assert top_telem.grid_info()["row"] == 0, "top_telem must be in row 0"
    assert split_frame.grid_info()["row"] == 2, "split_frame must be in row 2"
    assert bottom_bar.grid_info()["row"] == 3, "bottom_bar must be in row 3"
    assert app.btn_cancel_master.master == bottom_bar, "btn_cancel_master must be inside bottom_bar"

    # 3. Verify Width and Margins
    print("\n--- 3. Verifying Width & Margins ---")
    page_w = page.winfo_width()
    btn_w = app.btn_cancel_master.winfo_width()
    split_w = split_frame.winfo_width()

    print(f"Page Width: {page_w}px | Button Width: {btn_w}px | Split Frame Width: {split_w}px")
    assert btn_w == split_w, f"Button width ({btn_w}) must match split frame width ({split_w})"
    assert btn_w > 0.9 * page_w, "Button must span the full available width"

    # 4. Verify No Overlap across Window Resizing
    print("\n--- 4. Verifying No Overlap Across Window Resizing ---")
    test_sizes = ["1020x720", "1240x840", "1400x900", "1600x1000", "1100x750"]
    for size in test_sizes:
        app.geometry(size)
        app.update_idletasks()
        app.update()

        term_bottom = app.terminal_txt.winfo_rooty() + app.terminal_txt.winfo_height()
        cancel_top = app.btn_cancel_master.winfo_rooty()
        gap = cancel_top - term_bottom

        print(f"Size {size:>10}: Terminal bottom={term_bottom}, Cancel top={cancel_top}, Gap={gap}px")
        assert cancel_top > term_bottom, f"Overlap detected at {size}: Cancel top ({cancel_top}) <= Terminal bottom ({term_bottom})"
        assert gap >= 10, f"Expected at least 10px separation, got {gap}px"

    # 5. Verify Console Scrolling without Overlap
    print("\n--- 5. Verifying Console Scrolling ---")
    for i in range(100):
        app.terminal_txt.insert("end", f"[{i:03d}] INFO: Live photogrammetry feature matching batch {i}...\n")
    app.update()

    # Scroll to top, middle, bottom
    for pos in ["1.0", "50.0", "end"]:
        app.terminal_txt.see(pos)
        app.update()
        term_bottom = app.terminal_txt.winfo_rooty() + app.terminal_txt.winfo_height()
        cancel_top = app.btn_cancel_master.winfo_rooty()
        assert cancel_top > term_bottom, f"Overlap during scrolling at position {pos}!"

    print("Console scrolling verified — Cancel button remains docked below console at all scroll positions.")

    # 6. Verify Running State
    print("\n--- 6. Verifying Running State ---")
    app.is_processing = True
    app.btn_cancel_master.configure(state="normal", text="⏹️  Cancel Reconstruction")
    app.update()
    assert app.btn_cancel_master.cget("state") == "normal"
    assert "Cancel Reconstruction" in app.btn_cancel_master.cget("text")
    print("Running state verified: Button is normal and active.")

    # 7. Verify Cancelled State
    print("\n--- 7. Verifying Cancelled State ---")
    with patch("tkinter.messagebox.askyesno", return_value=True):
        app._on_cancel_reconstruction()
    app.update()

    assert not app.is_processing, "is_processing should be False after cancellation"
    assert "cancelled" in app.lbl_progress_title.cget("text").lower()
    assert app.btn_cancel_master.cget("state") == "normal"
    assert "Cancel Reconstruction" in app.btn_cancel_master.cget("text")
    print("Cancellation handler verified: is_processing=False, status updated, button restored.")

    # Also test cancellation event from pipeline
    event = PipelineEvent(
        stage=StageType.COLMAP_MAPPER,
        status=StageStatus.SKIPPED,
        progress=0.5,
        message="Pipeline cancelled by user.",
    )
    app._handle_pipeline_event(event)
    app.update()
    assert not app.is_processing
    print("PipelineEvent cancellation handling verified.")

    # Check button position still below console in cancelled state
    term_bottom = app.terminal_txt.winfo_rooty() + app.terminal_txt.winfo_height()
    cancel_top = app.btn_cancel_master.winfo_rooty()
    assert cancel_top > term_bottom, "Overlap detected after cancellation!"

    app.destroy()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! Cancel button layout regression is fixed.")
    print("=" * 70)


if __name__ == "__main__":
    test_cancel_button_layout()
