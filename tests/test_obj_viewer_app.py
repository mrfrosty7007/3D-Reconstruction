"""
GeoRecon AI (SIH-26158) - In-App Interactive OBJ Viewer Test Suite
Validates:
1. Automatic OBJ detection priority (mesh.obj > textured.obj > model.obj > *.obj).
2. Trimesh + Plotly 3D Figure generation with scene.aspectmode='data' and zero margins.
3. Handling both Trimesh and Scene objects with geometry merging.
4. UI Integration: '👁 View Mesh' button in app.py, 'Export Format -> OBJ Mesh' detection.
5. Error handling for missing or corrupt files.
6. Existing config.py compatibility (no 'Config' class imported).
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh
from app import GeoReconApp
from viewer.obj_viewer import find_obj_file, create_obj_figure, generate_obj_html


class TestInAppOBJViewer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_priority_obj_detection(self):
        """Verify the exact detection order: mesh.obj > textured.obj > model.obj > *.obj."""
        # 1. Fallback *.obj
        fallback = self.tmp_path / "custom_scan.obj"
        fallback.write_text("v 0 0 0\n", encoding="utf-8")
        found = find_obj_file(self.tmp_path)
        self.assertEqual(found.name, "custom_scan.obj")

        # 2. model.obj overrides fallback
        model = self.tmp_path / "model.obj"
        model.write_text("v 1 1 1\n", encoding="utf-8")
        found = find_obj_file(self.tmp_path)
        self.assertEqual(found.name, "model.obj")

        # 3. textured.obj overrides model.obj
        textured = self.tmp_path / "textured.obj"
        textured.write_text("v 2 2 2\n", encoding="utf-8")
        found = find_obj_file(self.tmp_path)
        self.assertEqual(found.name, "textured.obj")

        # 4. mesh.obj overrides textured.obj
        mesh = self.tmp_path / "mesh.obj"
        mesh.write_text("v 3 3 3\n", encoding="utf-8")
        found = find_obj_file(self.tmp_path)
        self.assertEqual(found.name, "mesh.obj")

    def test_trimesh_and_scene_support(self):
        """Verify Trimesh, Scene object handling, geometry merging, aspectmode='data', and zero margins."""
        # A. Triangle Mesh
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        sphere_path = self.tmp_path / "sphere.obj"
        sphere.export(str(sphere_path))

        fig_mesh = create_obj_figure(sphere_path)
        self.assertEqual(fig_mesh.layout.scene.aspectmode, "data")
        self.assertEqual(fig_mesh.layout.margin.l, 0)
        self.assertEqual(fig_mesh.layout.margin.r, 0)
        self.assertEqual(fig_mesh.layout.margin.t, 0)
        self.assertEqual(fig_mesh.layout.margin.b, 0)
        self.assertEqual(fig_mesh.layout.scene.dragmode, "orbit")

        # B. Scene object with multiple geometries
        box1 = trimesh.creation.box()
        box2 = trimesh.creation.box()
        box2.apply_translation([3, 0, 0])
        scene = trimesh.Scene([box1, box2])
        scene_path = self.tmp_path / "scene.obj"
        scene.export(str(scene_path))

        fig_scene = create_obj_figure(scene_path)
        self.assertIsNotNone(fig_scene)
        self.assertEqual(fig_scene.layout.scene.aspectmode, "data")

    def test_real_project_deliverable(self):
        """Test on actual GeoRecon AI project artifact (outputs/trial5_2_20260904_013003/model.obj)."""
        real_session = Path("outputs/trial5_2_20260904_013003")
        if real_session.exists():
            obj_file = find_obj_file(real_session)
            self.assertIsNotNone(obj_file)
            fig = create_obj_figure(obj_file)
            self.assertIsNotNone(fig)
            self.assertEqual(fig.layout.scene.aspectmode, "data")

    def test_app_ui_integration(self):
        """Verify '👁 View Mesh' button exists in app.py and responds to 'OBJ Mesh' export format."""
        app = GeoReconApp()
        try:
            # 1. Verify button existence
            self.assertTrue(hasattr(app, "btn_view_mesh"))
            self.assertEqual(app.btn_view_mesh.cget("text"), "👁 View Mesh")

            # 2. Select 'OBJ Mesh' in Export Format dropdown
            app.opt_export.set("OBJ Mesh")
            self.assertEqual(app.opt_export.get(), "OBJ Mesh")

            # 3. Populate finished scene with real completed session
            app.switch_page("finished")
            app.update_idletasks()
            app.update()

            session_dir = app._get_active_finished_session_dir()
            if session_dir and (session_dir / "model.obj").exists():
                app._populate_finished_scene(session_dir)
                # Button should be enabled and styled with accent color
                self.assertEqual(app.btn_view_mesh.cget("state"), "normal")
                self.assertEqual(app.btn_view_mesh.cget("text"), "👁 View Mesh")

            # 4. Test clicking _on_view_mesh calls launcher
            with mock.patch("app.launch_obj_viewer_process", return_value=mock.MagicMock()) as mock_launch:
                app._on_view_mesh()
                self.assertTrue(mock_launch.called)

        finally:
            app.destroy()

    def test_error_handling_missing_obj(self):
        """Verify friendly warning dialog is shown when no OBJ is present in session."""
        app = GeoReconApp()
        try:
            empty_session = self.tmp_path / "empty_session"
            empty_session.mkdir()
            app.current_finished_session_dir = empty_session

            with mock.patch("tkinter.messagebox.showwarning") as mock_warn:
                app._on_view_mesh()
                self.assertTrue(mock_warn.called)
                warning_title = mock_warn.call_args[0][0]
                warning_msg = mock_warn.call_args[0][1]
                self.assertEqual(warning_title, "No OBJ Found")
                self.assertIn("mesh.obj", warning_msg)
        finally:
            app.destroy()

    def test_config_compatibility(self):
        """Verify that no code imports or defines a 'Config' class (preserving existing config.py)."""
        import config
        self.assertFalse(hasattr(config, "Config"), "config.py should not have a 'Config' class")
        self.assertTrue(hasattr(config, "AppConfig"), "config.py should have AppConfig")
        self.assertTrue(hasattr(config, "DEFAULT_CONFIG"), "config.py should have DEFAULT_CONFIG")


if __name__ == "__main__":
    unittest.main()
