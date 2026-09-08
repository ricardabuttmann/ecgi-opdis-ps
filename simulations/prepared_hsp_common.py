"""Shared openCARP setup for the public prepared HSP simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from carputils import tools

from opdis_ps.simulation import load_simulation_config
from opdis_ps.simulation_case import PreparedSimulationCase, load_prepared_simulation_case


CAP_TAG = 300


@dataclass(frozen=True)
class StimulusSpec:
    stim_type: int
    strength: float
    duration: float
    start: float
    vtx_file: Path


class CarputilsAdapter(Protocol):
    @staticmethod
    def parameter_file(path: Path) -> Any: ...

    @staticmethod
    def stimulus(name: str, spec: StimulusSpec) -> Any: ...


def parser():
    argument_parser = tools.standard_parser()
    argument_parser.add_argument("--case-dir", type=Path, required=True)
    argument_parser.add_argument("--config", type=Path, required=True)
    argument_parser.add_argument("--output", type=Path, required=True)
    argument_parser.add_argument("--seed", type=int, required=True)
    return argument_parser


def job_id(args) -> str:
    return str(args.output.resolve())


def _element_tags(path: Path, expected_elements: int) -> list[int]:
    tags = set()
    with path.open(encoding="utf-8") as handle:
        try:
            declared = int(handle.readline())
        except ValueError as error:
            raise ValueError(f"{path}: invalid element count") from error
        if declared != expected_elements:
            raise ValueError(f"{path}: {declared} elements != expected {expected_elements}")
        observed = 0
        for observed, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) < 6 or fields[0] != "Tt":
                raise ValueError(f"{path}: only tagged tetrahedral Tt elements are supported")
            tags.add(int(fields[-1]))
    if observed != declared:
        raise ValueError(f"{path}: header says {declared} elements but read {observed}")
    return sorted(tags)


def _mesh_base(case: PreparedSimulationCase) -> Path:
    points = case.path("mesh_points")
    base = points.with_suffix("")
    expected = {
        "mesh_points": base.with_suffix(".pts"),
        "mesh_elements": base.with_suffix(".elem"),
        "mesh_fibers": base.with_suffix(".lon"),
    }
    for role, path in expected.items():
        if case.path(role) != path:
            raise ValueError("mesh_points, mesh_elements, and mesh_fibers must share one basename")
    return base


def write_run_info(
    *,
    output: Path,
    case: PreparedSimulationCase,
    settings: dict[str, Any],
    myocardial_tags: list[int],
    passive_tags: list[int],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    lines = [
        "Prepared pseudo-bidomain simulation",
        f"case_id: {case.case_id}",
        f"seed: {case.seed}",
        f"bidomain_mode: {settings['bidomain_mode']}",
        f"tend_ms: {settings['tend_ms']}",
        f"spacedt_ms: {settings['spacedt_ms']}",
        f"timedt_ms: {settings['timedt_ms']}",
        f"myocardial_tags: {myocardial_tags}",
        f"passive_tags: {passive_tags}",
        "stimuli:",
    ]
    lines.extend(
        f"  {site.name}: {site.start_ms} ms ({case.files[site.file_role].path})"
        for site in case.stimuli
    )
    (output / "run_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_hsp(args, job, *, adapter: CarputilsAdapter) -> None:
    case = load_prepared_simulation_case(args.case_dir)
    settings = load_simulation_config(args.config)
    if args.seed != case.seed:
        raise ValueError(f"seed {args.seed} does not match prepared seed {case.seed}")

    mesh_base = _mesh_base(case)
    tags = _element_tags(case.path("mesh_elements"), case.expected["volume_elements"])
    myocardial_tags = [tag for tag in tags if tag >= 100 and tag != CAP_TAG]
    passive_tags = [tag for tag in tags if tag < 100 or tag == CAP_TAG]
    if not myocardial_tags or not passive_tags:
        raise ValueError("prepared mesh must contain both myocardial and passive region tags")

    command = tools.carp_cmd(adapter.parameter_file(case.path("parameter_file")))
    command += [
        "-simID",
        job.ID,
        "-meshname",
        str(mesh_base),
        "-tend",
        settings["tend_ms"],
        "-bidomain",
        settings["bidomain_mode"],
        "-spacedt",
        settings["spacedt_ms"],
        "-timedt",
        settings["timedt_ms"],
    ]
    command += tools.gen_physics_opts(IntraTags=myocardial_tags, ExtraTags=tags)

    stimuli = [
        adapter.stimulus(
            f"S1_{site.name}",
            StimulusSpec(
                stim_type=0,
                strength=float(settings["stim_strength"]),
                duration=float(settings["stim_duration_ms"]),
                start=site.start_ms,
                vtx_file=case.path(site.file_role),
            ),
        )
        for site in case.stimuli
    ]
    stimuli.append(
        adapter.stimulus(
            "ground",
            StimulusSpec(
                stim_type=3,
                strength=0.0,
                duration=float(settings["tend_ms"]),
                start=0.0,
                vtx_file=case.path("ground_vertices"),
            ),
        )
    )
    command += ["-num_stim", len(stimuli)]
    for index, stimulus in enumerate(stimuli):
        command += stimulus.opts_formatted(index)

    write_run_info(
        output=Path(job.ID),
        case=case,
        settings=settings,
        myocardial_tags=myocardial_tags,
        passive_tags=passive_tags,
    )
    job.carp(command)
