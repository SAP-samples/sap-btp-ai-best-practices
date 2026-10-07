"""Permanent, transactional deletion of snapshots and terminal run history."""

ACTIVE = frozenset({"queued", "running", "persisting"})


class RemovalService:
    """Delete owned HANA records while serializing against job submission/publication."""

    def remove_run(self, run_id):
        """Delete a terminal run ID, result rows and artifacts; return deletion acknowledgement."""
        with self.repo.transaction():
            value = self.repo.get("runs", run_id)
            if value["status"] in ACTIVE:
                raise ValueError("Cancel the active run and wait for termination before deleting it")
            # Acquire the same revision lock used by workers before removing data.
            self.repo.cas("runs", run_id, value["revision"], {})
            self.repo.delete_owner("runs", run_id)
        return {"run_id": run_id, "deleted": True}

    def remove_dataset(self, dataset_id):
        """Delete a snapshot and all dependent drafts/terminal runs atomically; reject active jobs."""
        with self.repo.transaction():
            value = self.repo.get("datasets", dataset_id)
            # Lock before discovery so a concurrent launch cannot add an unseen job.
            self.repo.cas("datasets", dataset_id, value["revision"], {})
            runs = self.repo.list("runs", {"dataset_id": dataset_id})
            if any(run["status"] in ACTIVE for run in runs):
                raise ValueError("This snapshot has active runs; cancel or finish them before deletion")
            for run in runs:
                self.remove_run(run["run_id"])
            for draft in self.repo.list("drafts", {"dataset_id": dataset_id}):
                self.repo.delete_owner("drafts", draft["draft_id"])
            self.repo.delete_owner("datasets", dataset_id)
        return {"dataset_id": dataset_id, "deleted": True, "deleted_runs": len(runs)}
