from __future__ import annotations

from app.repositories.jobs_repository import JobsRepository


def _make_repo() -> JobsRepository:
    return JobsRepository()


class TestJobsRepository:
    def test_get_status_returns_none_for_unknown_job(self):
        repo = _make_repo()
        assert repo.get_status("nonexistent") is None

    def test_set_and_get_status_cycle(self):
        repo = _make_repo()
        repo.set_status("job1", "processing", 50, "Procesando...")
        status = repo.get_status("job1")
        assert status is not None
        assert status.job_id == "job1"
        assert status.status == "processing"
        assert status.progress == 50
        assert status.message == "Procesando..."
        assert status.song is None

    def test_set_status_clamps_progress_above_100(self):
        repo = _make_repo()
        repo.set_status("job1", "processing", 150, "msg")
        assert repo.get_status("job1").progress == 100

    def test_set_status_clamps_progress_below_0(self):
        repo = _make_repo()
        repo.set_status("job1", "processing", -5, "msg")
        assert repo.get_status("job1").progress == 0

    def test_clear_removes_job(self):
        repo = _make_repo()
        repo.set_status("job1", "completed", 100, "Listo")
        repo.clear("job1")
        assert repo.get_status("job1") is None

    def test_clear_nonexistent_job_does_not_raise(self):
        repo = _make_repo()
        repo.clear("ghost-job")

    def test_overwrite_existing_job(self):
        repo = _make_repo()
        repo.set_status("job1", "processing", 30, "Iniciando")
        repo.set_status("job1", "completed", 100, "Listo")
        status = repo.get_status("job1")
        assert status.status == "completed"
        assert status.progress == 100
