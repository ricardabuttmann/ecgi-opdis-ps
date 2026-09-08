#!/usr/bin/env python
"""Prepared HSP simulation runner for openCARP/carputils 11."""

from pathlib import Path

from carputils import tools
from carputils.model import Stimulus
from prepared_hsp_common import StimulusSpec, job_id, parser, run_hsp


class V11CarputilsAdapter:
    @staticmethod
    def parameter_file(path: Path) -> str:
        return str(path)

    @staticmethod
    def stimulus(name: str, spec: StimulusSpec) -> Stimulus:
        return Stimulus(
            name,
            stimtype=spec.stim_type,
            strength=spec.strength,
            duration=spec.duration,
            start=spec.start,
            vtx_file=str(spec.vtx_file),
        )


@tools.carpexample(parser, job_id)
def run(args, job):
    run_hsp(args, job, adapter=V11CarputilsAdapter)


if __name__ == "__main__":
    run()
