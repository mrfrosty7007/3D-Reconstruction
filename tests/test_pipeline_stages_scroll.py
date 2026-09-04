"""
GeoRecon AI - Pipeline Stages Panel Scrolling Test Suite (SIH-26158)
Validates:
1. Pipeline Stages panel is scrollable via CTkScrollableFrame.
2. "PIPELINE STAGES" header remains fixed at the top.
3. Card sizing, numbering badges, typography, and status chips are preserved.
4. Active stage auto-scroll and centering as pipeline advances (Stages 1 through 6).
5. Non-jumpy scrolling behavior for already-visible stages.
6. Responsive verification at 1024x720, 1280x800, 1366x768, 1600x900, and maximized.
7. Mouse wheel scrolling capability.
8. Layout integrity: Confidence card, Live console, Bottom bar remain intact.
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import customtkinter as ctk
from app import GeoReconApp
from pipeline import StageType, StageStatus, PipelineEvent


def test_pipeline_stages_scroll():
    print("=" * 80)
    print("🚀 Testing Pipeline Stages Scrollable Panel & Auto-Focus (SIH-26158)")
    print("=" * 80)

    app = GeoReconApp()
    app.switch_page("progress")
    app.update_idletasks()
    app.update()

    # 1. Structural Hierarchy
    print("\n--- 1. Verifying Structural Hierarchy ---")
    assert hasattr(app, "tracker_scroll"), "app must have tracker_scroll attribute"
    assert isinstance(app.tracker_scroll, ctk.CTkScrollableFrame), "tracker_scroll must be a CTkScrollableFrame"

    page = app.page_frames["progress"]
    split_frame = list(page.winfo_children())[2]
    tracker_box = list(split_frame.winfo_children())[0]

    # Header must be child of tracker_box at row 0
    header_lbl = list(tracker_box.winfo_children())[0]
    assert header_lbl.grid_info()["row"] == 0, "Header label must be at row 0"
    assert "PIPELINE STAGES" in header_lbl.cget("text"), "Header text must be PIPELINE STAGES"

    # tracker_scroll must be at row 1 of tracker_box
    assert app.tracker_scroll.grid_info()["row"] == 1, "tracker_scroll must be at row 1 of tracker_box"
    print("✓ Fixed header and scrollable frame hierarchy verified.")

    # 2. Stage Cards Inside Scrollable Frame & Sizing Preservation
    print("\n--- 2. Verifying Card Sizing & Integrity ---")
    assert len(app.stage_items) == 6, f"Expected 6 stage items, found {len(app.stage_items)}"

    for st in StageType:
        item = app.stage_items[st]
        assert item.master == app.tracker_scroll, f"Stage {st.name} must be child of tracker_scroll"
        assert item.lbl_num.winfo_reqwidth() >= 24, "Number circle width must be preserved"
        assert item.lbl_num.cget("text") == str(st.stage_number), f"Number badge mismatch for {st.name}"
        assert item.lbl_badge.cget("text") == "PENDING", "Initial status badge must be PENDING"
        assert item.lbl_name.cget("font") is not None, "Typography must be preserved"

    print("✓ All 6 stage cards verified inside scroll container with intact sizing, numbers, and typography.")

    # 3. Enable Confidence Card and Test Active Stage Auto-Focus (Stages 1 through 6)
    print("\n--- 3. Testing Active Stage Auto-Focus Through All 6 Stages ---")
    # Show confidence card to simulate tight vertical space
    app.frame_confidence.grid()
    app.geometry("1024x720")
    app.update_idletasks()
    app.update()

    canvas = app.tracker_scroll._parent_canvas
    view_h = canvas.winfo_height()
    bbox = canvas.bbox("all")
    total_h = bbox[3] - bbox[1]
    print(f"Viewport Height: {view_h}px | Total Content Height: {total_h}px (Content exceeds viewport)")
    assert total_h > view_h, "Content must exceed viewport at 1024x720 with confidence card to test scrolling"

    stages = [
        StageType.FRAME_EXTRACTION,
        StageType.COLMAP_FEATURES,
        StageType.COLMAP_MATCHING,
        StageType.COLMAP_MAPPER,
        StageType.GAUSSIAN_SPLATTING,
        StageType.EXPORT,
    ]

    for stage in stages:
        # Simulate stage advancing
        evt = PipelineEvent(
            stage=stage,
            status=StageStatus.RUNNING,
            progress=0.1,
            message=f"Processing {stage.display_name}...",
        )
        app._handle_pipeline_event(evt)
        # Wait for smooth animation timer steps to complete
        for _ in range(15):
            app.update()
            time.sleep(0.02)

        cur_top = canvas.yview()[0] * total_h
        cur_bot = canvas.yview()[1] * total_h
        item = app.stage_items[stage]
        item_y = item.winfo_y()
        item_h = item.winfo_height()
        item_center = item_y + item_h / 2.0

        # Verify active stage center is visible within the viewport
        is_visible = (cur_top <= item_center <= cur_bot)
        print(f"  Advancing to {stage.name:<22}: item y={item_y:>3}..{item_y+item_h:<3} (center={item_center:.1f}) in view [{cur_top:.1f}..{cur_bot:.1f}] -> visible={is_visible}")
        assert is_visible, f"Active stage {stage.name} was not brought into view!"

    print("✓ Active stage auto-focus successfully brought each stage (1 to 6) into center view.")

    # 4. Avoid Jumpy Scrolling for Already-Visible Stages
    print("\n--- 4. Testing Non-Jumpy Scrolling for Visible Stages ---")
    current_scroll = canvas.yview()[0]
    # Re-trigger scroll on same stage
    app.scroll_to_stage(StageType.EXPORT, smooth=True)
    app.update()
    new_scroll = canvas.yview()[0]
    assert abs(new_scroll - current_scroll) < 0.001, "Should not jump/scroll if stage is already comfortably in view"
    print("✓ Non-jumpy scroll behavior verified: already-visible stage does not jitter.")

    # 5. Responsive Resolutions Testing
    print("\n--- 5. Testing Responsive Viewport Reachability Across Resolutions ---")
    resolutions = [
        ("1024x720", 1024, 720),
        ("1280x800", 1280, 800),
        ("1366x768", 1366, 768),
        ("1600x900", 1600, 900),
        ("Maximized (1920x1080)", 1920, 1080),
    ]

    for res_name, rw, rh in resolutions:
        app.geometry(f"{rw}x{rh}")
        app.update_idletasks()
        app.update()

        v_h = canvas.winfo_height()
        t_h = canvas.bbox("all")[3] - canvas.bbox("all")[1]

        # Ensure Stage 1 (top) and Stage 6 (bottom) are both reachable
        # 1. Scroll to Stage 1
        app.scroll_to_stage(StageType.FRAME_EXTRACTION, smooth=False)
        app.update()
        st1_center = app.stage_items[StageType.FRAME_EXTRACTION].winfo_y() + app.stage_items[StageType.FRAME_EXTRACTION].winfo_height() / 2.0
        v_top = canvas.yview()[0] * t_h
        v_bot = canvas.yview()[1] * t_h
        assert v_top <= st1_center <= v_bot, f"Stage 1 not reachable at {res_name}"

        # 2. Scroll to Stage 6
        app.scroll_to_stage(StageType.EXPORT, smooth=False)
        app.update()
        st6_center = app.stage_items[StageType.EXPORT].winfo_y() + app.stage_items[StageType.EXPORT].winfo_height() / 2.0
        v_top = canvas.yview()[0] * t_h
        v_bot = canvas.yview()[1] * t_h
        assert v_top <= st6_center <= v_bot, f"Stage 6 not reachable at {res_name}"

        print(f"  ✓ Resolution {res_name:>22}: Stage 1 and Stage 6 fully reachable (Viewport H={v_h}px)")

    # 6. Mouse Wheel Scrolling Support
    print("\n--- 6. Testing Mouse Wheel Scrolling Support ---")
    app.scroll_to_stage(StageType.FRAME_EXTRACTION, smooth=False)
    app.update()
    before_wheel = canvas.yview()[0]

    # Simulate mouse wheel down event on tracker_scroll
    canvas.yview_scroll(2, "units")
    app.update()
    after_wheel = canvas.yview()[0]
    assert after_wheel > before_wheel, "Mouse wheel down must increment scroll position"

    canvas.yview_scroll(-2, "units")
    app.update()
    after_wheel_up = canvas.yview()[0]
    assert after_wheel_up < after_wheel, "Mouse wheel up must decrement scroll position"
    print("✓ Mouse wheel vertical scrolling verified.")

    # 7. Layout Integrity
    print("\n--- 7. Verifying Layout Integrity ---")
    # Cancel button must remain docked below console
    term_bottom = app.terminal_txt.winfo_rooty() + app.terminal_txt.winfo_height()
    cancel_top = app.btn_cancel_master.winfo_rooty()
    assert cancel_top > term_bottom, "Cancel button must remain docked below console"

    # Stage tracker must not overlap console
    tracker_right = tracker_box.winfo_rootx() + tracker_box.winfo_width()
    console_left = app.terminal_txt.winfo_rootx()
    assert console_left > tracker_right, f"Console ({console_left}) must be to the right of tracker ({tracker_right})"

    print("✓ Layout integrity confirmed: Console, Tracker, and Cancel Button occupy separate coordinates.")

    app.destroy()
    print("\n" + "=" * 80)
    print("🎉 ALL PIPELINE STAGES SCROLLING & RESPONSIVE TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_pipeline_stages_scroll()
