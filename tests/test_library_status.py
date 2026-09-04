# GeoRecon AI - Phase 6.3 Test Suite
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import GeoReconApp, DEFAULT_CONFIG
from pipeline.manager import (
    PipelineManager,
    infer_session_status,
    PIPELINE_STATUS_COMPLETED,
    PIPELINE_STATUS_FAILED,
    PIPELINE_STATUS_PARTIAL,
    PIPELINE_STATUS_CANCELLED,
)

def run_tests():
    print('=' * 80)
    print('GeoRecon AI -- Phase 6.3: Model Library Status and Failed Session Handling')
    print('=' * 80)

    with tempfile.TemporaryDirectory(prefix='georecon_test_lib_') as tmp_dir:
        tmp_path = Path(tmp_dir)
        outputs_dir = tmp_path / 'outputs'
        data_dir = tmp_path / 'data'
        outputs_dir.mkdir()
        data_dir.mkdir()

        mgr = PipelineManager()

        # 1. Single Source of Truth
        print('\n--- 1. Testing scene_manifest.json Single Source of Truth ---')
        session_success = outputs_dir / 'session_success_test'
        session_failed = outputs_dir / 'session_failed_gate_test'
        session_cancelled = outputs_dir / 'session_cancelled_test'
        session_partial = outputs_dir / 'session_partial_test'

        mgr.write_scene_manifest(
            session_output_dir=session_success,
            session_name='session_success_test',
            pipeline_status=PIPELINE_STATUS_COMPLETED,
            pipeline_stage_completed=6,
            quality_gate_passed=True,
            registered_cameras=130,
            total_cameras=130,
            sparse_points=35904,
            extra_data={
                'psnr': 33.4,
                'gaussians': 56173,
                'video': {'name': 'survey.mp4'},
            },
        )
        (session_success / 'point_cloud.ply').write_text('ply\nformat ascii 1.0\nelement vertex 35904\nend_header\n')
        (session_success / 'trajectory_preview.mp4').write_bytes(b'fake_mp4_bytes')
        (session_success / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (session_success / 'checkpoints' / 'checkpoint_final.json').write_text(json.dumps({'final_psnr': 33.4, 'gaussian_count': 56173}))

        mgr.write_scene_manifest(
            session_output_dir=session_failed,
            session_name='session_failed_gate_test',
            pipeline_status=PIPELINE_STATUS_FAILED,
            pipeline_stage_completed=4,
            quality_gate_passed=False,
            registered_cameras=10,
            total_cameras=85,
            registration_percentage=11.8,
            sparse_points=3762,
            failure_reason='Quality Gate Failed: Only 10/85 cameras registered (11.8%).',
        )
        (session_failed / 'colmap_summary.json').write_text(json.dumps({
            'session_name': 'session_failed_gate_test',
            'registered_cameras': 10,
            'total_cameras': 85,
            'registration_percentage': 11.8,
            'sparse_point_count': 3762,
            'proceed': False,
            'quality_gate_passed': False,
            'status': 'FAILED',
        }))
        (session_failed / 'recovery_suggestions.json').write_text(json.dumps({
            'quality_level': 'RED',
            'registered_percentage': 11.8,
            'registered_cameras': 10,
            'total_cameras': 85,
            'sparse_points': 3762,
        }))

        mgr.write_scene_manifest(
            session_output_dir=session_cancelled,
            session_name='session_cancelled_test',
            pipeline_status=PIPELINE_STATUS_CANCELLED,
            pipeline_stage_completed=2,
            quality_gate_passed=False,
            registered_cameras=0,
            total_cameras=85,
            failure_reason='Reconstruction cancelled by user.',
        )

        mgr.export_partial_session(
            session_output_dir=session_partial,
            session_name='session_partial_test',
            total_cameras=85,
            registered_cameras=60,
            sparse_points=12000,
            quality_gate_passed=True,
        )

        m_succ = json.loads((session_success / 'scene_manifest.json').read_text(encoding='utf-8'))
        assert m_succ['pipeline_status'] == 'completed'
        assert m_succ['pipeline_stage_completed'] == 6
        assert m_succ['quality_gate_passed'] is True
        assert m_succ['registered_cameras'] == 130
        assert m_succ['total_cameras'] == 130

        m_fail = json.loads((session_failed / 'scene_manifest.json').read_text(encoding='utf-8'))
        assert m_fail['pipeline_status'] == 'failed'
        assert m_fail['pipeline_stage_completed'] == 4
        assert m_fail['quality_gate_passed'] is False
        assert m_fail['registered_cameras'] == 10
        assert m_fail['total_cameras'] == 85
        assert 'Quality Gate Failed' in m_fail['failure_reason']

        print('Manifest schema and writer verified with 5 required fields.')

        # 2. Status inference & Backward compatibility
        print('\n--- 2. Testing Status Inference and Backward Compatibility ---')
        assert infer_session_status(session_success) == 'completed'
        assert infer_session_status(session_failed) == 'failed'
        assert infer_session_status(session_cancelled) == 'cancelled'
        assert infer_session_status(session_partial) == 'partial'

        session_leg_comp = outputs_dir / 'session_legacy_completed'
        session_leg_comp.mkdir()
        (session_leg_comp / 'point_cloud.ply').write_text('ply content')
        (session_leg_comp / 'trajectory_preview.mp4').write_bytes(b'mp4')
        (session_leg_comp / 'checkpoints').mkdir()
        (session_leg_comp / 'checkpoints' / 'checkpoint_final.json').write_text('{}')
        assert infer_session_status(session_leg_comp) == 'completed'

        session_leg_fail = outputs_dir / 'session_legacy_failed'
        session_leg_fail.mkdir()
        (session_leg_fail / 'colmap_summary.json').write_text(json.dumps({
            'proceed': False,
            'registered_cameras': 5,
            'total_cameras': 100,
            'registration_percentage': 5.0,
        }))
        assert infer_session_status(session_leg_fail) == 'failed'

        session_leg_part = outputs_dir / 'session_legacy_partial'
        session_leg_part.mkdir()
        (session_leg_part / 'point_cloud.ply').write_text('ply')
        assert infer_session_status(session_leg_part) == 'partial'

        wa_path = Path('outputs/WhatsApp_Video_2026-09-03_at_8.28.38_PM_20260903_211117')
        if wa_path.exists():
            st_wa = infer_session_status(wa_path)
            print(f'Existing WhatsApp Video Inferred Status: {st_wa}')
            assert st_wa == 'failed'

        print('Backward compatibility status inference verified.')

        # 3. GUI Model Library Cards
        print('\n--- 3. Testing Model Library GUI Cards and Action Logic ---')
        app = GeoReconApp()
        app.update_idletasks()

        app.config.outputs_dir = outputs_dir
        app.selected_output_dir = outputs_dir

        app.switch_page('library')
        app._refresh_model_library()
        app.update()

        cards = [child for child in app.library_scroll.winfo_children() if hasattr(child, 'session_status')]
        assert len(cards) >= 4

        card_map = {c.session_path.name: c for c in cards}

        c_succ = card_map['session_success_test']
        assert c_succ.badge_label.cget('text') == '🟢 COMPLETE'
        assert c_succ.cget('border_color') == '#202738'
        actions_succ = app.get_session_action_states(c_succ.session_path)
        assert actions_succ['view_3d'] is True
        assert actions_succ['preview'] is True
        assert actions_succ['export'] is True
        assert actions_succ['folder'] is True
        assert actions_succ['delete'] is True

        c_fail = card_map['session_failed_gate_test']
        assert c_fail.badge_label.cget('text') == '🔴 FAILED'
        assert c_fail.cget('border_color') == '#991B1B'
        assert hasattr(c_fail, 'fail_frame')
        actions_fail = app.get_session_action_states(c_fail.session_path)
        assert actions_fail['retry'] is True
        assert actions_fail['folder'] is True
        assert actions_fail['delete'] is True
        assert actions_fail['view_3d'] is False
        assert actions_fail['preview'] is False
        assert actions_fail['export'] is False
        assert c_fail.action_buttons['retry'].cget('state') == 'normal'
        assert c_fail.action_buttons['view_3d'].cget('state') == 'disabled'
        assert c_fail.action_buttons['preview'].cget('state') == 'disabled'
        assert c_fail.action_buttons['export'].cget('state') == 'disabled'
        assert c_fail.action_buttons['folder'].cget('state') == 'normal'
        assert c_fail.action_buttons['delete'].cget('state') == 'normal'

        c_canc = card_map['session_cancelled_test']
        assert c_canc.badge_label.cget('text') == '⚪ CANCELLED'

        c_part = card_map['session_partial_test']
        assert c_part.badge_label.cget('text') == '🟡 PARTIAL'

        print('Card badges, borders, and disabled button actions verified.')

        # 4. Finished Scene Guard
        print('\n--- 4. Testing Finished Scene Guard ---')
        # 4a. Explicit populate of completed session works
        app._populate_finished_scene(session_success)
        app.update()
        assert app.fin_card_scene.cget('text') == 'session_success_test'

        # 4b. Explicit populate of failed session is rejected by Guard
        app._populate_finished_scene(session_failed)
        app.update()
        displayed_scene = app.fin_card_scene.cget('text')
        assert displayed_scene != 'session_failed_gate_test', 'Security failure: Failed session appeared on Finished Scene!'
        # Fallback must be a completed session
        assert infer_session_status(outputs_dir / displayed_scene) == 'completed'
        print(f'  -> Finished Scene Guard successfully rejected failed session and displayed: {displayed_scene}')

        # 4c. When ONLY failed sessions exist
        with tempfile.TemporaryDirectory(prefix='georecon_only_failed_') as fail_only_tmp:
            fail_only_p = Path(fail_only_tmp)
            f_sess = fail_only_p / 'only_failed_sess'
            mgr.write_scene_manifest(f_sess, 'only_failed_sess', PIPELINE_STATUS_FAILED, 4, False)
            app.selected_output_dir = fail_only_p
            app.pipeline_mgr.last_session_name = 'only_failed_sess'
            app._populate_finished_scene()
            app.update()

            scene_lbl = app.fin_card_scene.cget('text')
            psnr_lbl = app.fin_card_psnr.cget('text')
            assert scene_lbl == 'No Completed Scene'
            assert psnr_lbl == '--'

        print('Finished Scene Guard verified: Only completed scenes can be displayed.')

        # 5. Retry Button Logic
        print('\n--- 5. Testing Retry Button Logic ---')
        app.selected_output_dir = outputs_dir
        app.switch_page('library')
        app._refresh_model_library()
        app.update()

        app._retry_session(session_failed)
        app.update()

        assert app.active_page_name == 'studio'
        assert app.entry_scene_name.get() == 'session_failed_gate_test'
        assert app.entry_blur.get() == '40.0'
        print('Retry button logic verified.')

        app.destroy()

    print('\n' + '=' * 80)
    print('ALL MODEL LIBRARY STATUS TESTS PASSED')
    print('=' * 80)

if __name__ == '__main__':
    run_tests()
