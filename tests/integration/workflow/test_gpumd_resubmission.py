import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from nepflow.errors import SchedulerError
from nepflow.mlip.gpumd import resubmission as module
from nepflow.workflow.resubmission import resolve_resubmit_command

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


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

    def test_segment_state_persists_timezone_aware_iso_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / module.STATE_FILE_NAME
            state = module.load_segment_state(state_path)
            module.write_segment_state(state_path, state)
            persisted = module.load_segment_state(state_path)

        for field in ("created", "updated"):
            parsed = datetime.fromisoformat(persisted[field])
            self.assertIsNotNone(parsed.tzinfo)
            self.assertIsNotNone(parsed.utcoffset())

    def test_gpumd_segment_uses_process_runner_boundary(self) -> None:
        runner = Mock()
        runner.run_shell.return_value.returncode = 0

        with tempfile.TemporaryDirectory() as tmp:
            module.run_gpumd_segment(
                "gpumd > log",
                Path(tmp),
                process_runner=runner,
            )

        runner.run_shell.assert_called_once_with(
            "gpumd > log",
            cwd=Path(tmp),
            check=False,
            capture_output=False,
        )

    def test_resolve_resubmit_command_falls_back_to_submit_script_in_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            submit_script = workdir / "submit.slurm"
            submit_script.write_text("#!/bin/bash\n", encoding="utf-8")

            command, cwd, source = resolve_resubmit_command(
                workdir=workdir,
                submit_script=None,
                environment={},
            )

            self.assertEqual(command, ["sbatch", str(submit_script)])
            self.assertEqual(cwd, workdir)
            self.assertEqual(source, f"submit.slurm in {workdir}")

    def test_root_utility_dry_run_is_a_supported_cli_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY_ROOT / "utilities" / "gpumd_self_resubmit.py"),
                    "--gpumd-command",
                    "fake-gpumd",
                    "--workdir",
                    str(workdir),
                    "--max-segments",
                    "1",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Would stop after the next segment", result.stdout)

    def test_scheduler_query_failure_does_not_fall_back_to_filesystem_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            submit_script = workdir / "submit.slurm"
            submit_script.write_text("#!/bin/bash\n", encoding="utf-8")
            scheduler = Mock()
            scheduler.show_job.side_effect = RuntimeError("scheduler unavailable")

            with pytest.raises(SchedulerError, match="scheduler unavailable") as error:
                resolve_resubmit_command(
                    workdir=workdir,
                    submit_script=None,
                    scheduler=scheduler,
                    environment={"SLURM_JOB_ID": "123"},
                )

            self.assertEqual(error.value.kind, "query_failed")
            scheduler.show_job.assert_called_once_with("123")

    def test_scheduler_lookup_reuses_original_submission_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            submit_script = workdir / "original.slurm"
            submit_script.write_text("#!/bin/bash\n", encoding="utf-8")
            scheduler = Mock()
            scheduler.show_job.return_value = f"JobId=123 Command={submit_script} WorkDir={workdir}"

            command, cwd, source = resolve_resubmit_command(
                workdir=workdir,
                scheduler=scheduler,
                environment={"SLURM_JOB_ID": "123"},
            )

            self.assertEqual(command, ["sbatch", str(submit_script)])
            self.assertEqual(cwd, workdir)
            self.assertEqual(source, "scontrol job 123")


if __name__ == "__main__":
    unittest.main()
