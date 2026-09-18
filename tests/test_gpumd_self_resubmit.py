import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
UTILITY = ROOT / "utilities" / "gpumd_self_resubmit.py"

spec = importlib.util.spec_from_file_location("gpumd_self_resubmit", UTILITY)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class GpumdSelfResubmitTests(unittest.TestCase):
    def test_archive_and_promote_final_keeps_history_and_updates_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            (workdir / "final.xyz").write_text("new structure\n", encoding="utf-8")
            (workdir / "model.xyz").write_text("old structure\n", encoding="utf-8")

            archived = module.archive_and_promote_final(
                workdir=workdir,
                archive_dir_name="history",
                final_name="final.xyz",
                model_name="model.xyz",
                segment_index=3,
            )

            archive_path = Path(archived["archived_final"])
            self.assertTrue(archive_path.exists())
            self.assertEqual(archive_path.read_text(encoding="utf-8"), "new structure\n")
            self.assertEqual((workdir / "model.xyz").read_text(encoding="utf-8"), "new structure\n")
            self.assertFalse((workdir / "final.xyz").exists())

    def test_should_stop_when_segment_limit_reached(self) -> None:
        stop, reason = module.should_stop(
            state={"segments_completed": 4},
            max_segments=4,
            stop_file=None,
        )
        self.assertTrue(stop)
        self.assertIn("max segments", reason)

    def test_resolve_resubmit_command_falls_back_to_submit_script_in_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            submit_script = workdir / "submit.slurm"
            submit_script.write_text("#!/bin/bash\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=False):
                command, cwd, source = module.resolve_resubmit_command(
                    workdir=workdir,
                    submit_script=None,
                    nepflow_root=None,
                )

            self.assertEqual(command, ["sbatch", str(submit_script)])
            self.assertEqual(cwd, workdir)
            self.assertEqual(source, f"submit.slurm in {workdir}")

    def test_main_runs_segment_updates_state_and_stops_at_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            final_file = workdir / "final.xyz"

            def fake_run(command: str, workdir: Path, dry_run: bool = False) -> None:
                self.assertEqual(command, "fake-gpumd")
                self.assertEqual(workdir, Path(tmp))
                final_file.write_text("segment output\n", encoding="utf-8")

            argv = [
                "gpumd_self_resubmit.py",
                "--gpumd-command",
                "fake-gpumd",
                "--workdir",
                str(workdir),
                "--max-segments",
                "1",
            ]

            with patch.object(module, "run_gpumd", side_effect=fake_run):
                with patch.object(module, "resubmit_job") as resubmit_mock:
                    with patch.object(module.sys, "argv", argv):
                        result = module.main()

            self.assertEqual(result, 0)
            resubmit_mock.assert_not_called()

            state = json.loads((workdir / module.STATE_FILE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(state["segments_completed"], 1)
            self.assertEqual(len(state["history"]), 1)
            self.assertTrue((workdir / "model.xyz").exists())
            self.assertTrue((workdir / module.DEFAULT_ARCHIVE_DIR / "segment_00001_final.xyz").exists())


if __name__ == "__main__":
    unittest.main()
