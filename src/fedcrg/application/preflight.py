"""Pre-training research preflight for frozen dataset and statistical contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fedcrg.application.precompute import ProtocolTablePrecomputer
from fedcrg.config.models import ExperimentConfig
from fedcrg.data.audit import PreparedDataAudit, PreparedDatasetAuditor


@dataclass(frozen=True, slots=True)
class PreflightResult:
    prepared_data: PreparedDataAudit
    readiness_table: Path
    mismatch_table: Path

    @property
    def valid(self) -> bool:
        return self.prepared_data.valid


class ResearchPreflight:
    """Require independently audited data and frozen statistical tables before training."""

    def __init__(
        self,
        data_auditor: PreparedDatasetAuditor | None = None,
        table_precomputer: ProtocolTablePrecomputer | None = None,
    ) -> None:
        self.data_auditor = data_auditor or PreparedDatasetAuditor()
        self.table_precomputer = table_precomputer or ProtocolTablePrecomputer()

    def run(self, config: ExperimentConfig, prepared_root: Path) -> PreflightResult:
        audit = self.data_auditor.audit(prepared_root, config)
        if not audit.valid:
            raise RuntimeError(
                "Prepared-data preflight failed: " + "; ".join(audit.problems)
            )
        readiness, mismatch = self.table_precomputer.precompute(config)
        return PreflightResult(
            prepared_data=audit,
            readiness_table=readiness,
            mismatch_table=mismatch,
        )
