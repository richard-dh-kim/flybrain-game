"""Rollout, metric, Parquet, and visual replay helpers for Phase 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import flybrain_game

from .curriculum import CurriculumEpisode, Trajectory
from .expert import InterceptExpertV1
from .schema import ObservationV1, synchronized_strike_is_ready


TRAJECTORY_SCHEMA_VERSION = 1
DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    episode_id: int
    trajectory: str
    status: int
    elapsed_ticks: int
    accepted_strikes: int
    completed_misses: int
    cooldown_violations: int
    closest_edge_gap_units: float

    @property
    def hit(self) -> bool:
        return self.status == 1


@dataclass(slots=True)
class Rollout:
    summary: EpisodeSummary
    rows: list[dict[str, Any]]


def _contact_gap_units(observation: ObservationV1) -> float:
    return min(
        math.hypot(
            hand.position[0] - observation.player_position[0],
            hand.position[1] - observation.player_position[1],
        )
        - flybrain_game.HAND_RADIUS_UNITS
        - flybrain_game.PLAYER_RADIUS_UNITS
        for hand in observation.hands
    )


def _observation_record(observation: ObservationV1) -> dict[str, Any]:
    record: dict[str, Any] = {
        "observation_schema_version": observation.schema_version,
        "tick": observation.tick,
        "player_x_units": observation.player_position[0],
        "player_y_units": observation.player_position[1],
        "player_velocity_x_units": observation.player_velocity[0],
        "player_velocity_y_units": observation.player_velocity[1],
        "player_acceleration_x_units": observation.player_acceleration[0],
        "player_acceleration_y_units": observation.player_acceleration[1],
        "attack_active": observation.attack_active,
        "round_status": observation.round_status,
        "round_elapsed_ticks": observation.round_elapsed_ticks,
        "round_remaining_ticks": observation.round_remaining_ticks,
    }
    for name, hand in zip(("left", "right"), observation.hands, strict=True):
        record.update(
            {
                f"{name}_hand_x_units": hand.position[0],
                f"{name}_hand_y_units": hand.position[1],
                f"{name}_hand_velocity_x_units": hand.velocity[0],
                f"{name}_hand_velocity_y_units": hand.velocity[1],
                f"{name}_hand_phase": hand.phase,
                f"{name}_hand_phase_tick": hand.phase_tick,
                f"{name}_hand_cooldown_remaining": hand.cooldown_remaining,
                f"{name}_hand_active_contact": hand.active_contact,
            }
        )
    return record


def run_episode(
    episode_id: int,
    name: str,
    trajectory: Trajectory,
    expert: InterceptExpertV1,
    *,
    maximum_ticks: int = 1_800,
    initial_player_position: tuple[int, int] | None = None,
) -> Rollout:
    simulation = (
        flybrain_game.Simulation()
        if initial_player_position is None
        else flybrain_game.Simulation.with_player_position(*initial_player_position)
    )
    rows: list[dict[str, Any]] = []
    accepted_strikes = 0
    completed_misses = 0
    cooldown_violations = 0
    closest_gap = math.inf

    while simulation.round_status == 0 and simulation.tick < maximum_ticks:
        observation = ObservationV1.from_simulation(simulation)
        destination = trajectory.destination(observation)
        action = expert.decide(observation)
        ready = synchronized_strike_is_ready(observation)
        strike_accepted = action.strike and ready
        cooldown_violation = action.strike and not ready
        if strike_accepted:
            accepted_strikes += 1
        if cooldown_violation:
            cooldown_violations += 1

        simulation.step_toward_with_action(
            destination[0],
            destination[1],
            action.target[0],
            action.target[1],
            action.strike,
        )
        post_observation = ObservationV1.from_simulation(simulation)
        completed_miss = (
            observation.attack_active
            and not post_observation.attack_active
            and post_observation.round_status == 0
        )
        if completed_miss:
            completed_misses += 1
        contact_gap = _contact_gap_units(post_observation)
        closest_gap = min(closest_gap, contact_gap)

        row = {
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
            "action_schema_version": action.schema_version,
            "episode_id": episode_id,
            "trajectory": name,
            **_observation_record(observation),
            "destination_x_units": destination[0],
            "destination_y_units": destination[1],
            "action_target_x_units": action.target[0],
            "action_target_y_units": action.target[1],
            "action_strike": action.strike,
            "strike_accepted": strike_accepted,
            "cooldown_violation": cooldown_violation,
            "post_round_status": post_observation.round_status,
            "post_attack_active": post_observation.attack_active,
            "contact_gap_units": contact_gap,
            "closest_edge_gap_units": closest_gap,
            "completed_miss": completed_miss,
            "hit": post_observation.round_status == 1,
        }
        rows.append(row)

    status = simulation.round_status
    elapsed_ticks = simulation.round_elapsed_ticks
    return Rollout(
        summary=EpisodeSummary(
            episode_id=episode_id,
            trajectory=name,
            status=status,
            elapsed_ticks=elapsed_ticks,
            accepted_strikes=accepted_strikes,
            completed_misses=completed_misses,
            cooldown_violations=cooldown_violations,
            closest_edge_gap_units=closest_gap,
        ),
        rows=rows,
    )


def evaluate(curriculum: Iterable[CurriculumEpisode], expert: InterceptExpertV1) -> list[Rollout]:
    return [
        run_episode(
            index,
            episode.name,
            episode.trajectory,
            expert,
            initial_player_position=episode.initial_player_position,
        )
        for index, episode in enumerate(curriculum)
    ]


def aggregate_metrics(rollouts: list[Rollout]) -> dict[str, Any]:
    summaries = [rollout.summary for rollout in rollouts]
    hits = [summary for summary in summaries if summary.hit]
    simple = [
        summary
        for summary in summaries
        if not summary.trajectory.startswith(("random-", "evasive"))
    ]
    simple_hits = sum(summary.hit for summary in simple)
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "observation_schema_version": flybrain_game.OBSERVATION_SCHEMA_VERSION,
        "action_schema_version": flybrain_game.ACTION_SCHEMA_VERSION,
        "episodes": len(summaries),
        "hits": len(hits),
        "hit_rate": len(hits) / len(summaries) if summaries else 0.0,
        "simple_episodes": len(simple),
        "simple_hits": simple_hits,
        "simple_hit_rate": simple_hits / len(simple) if simple else 0.0,
        "mean_hit_ticks": (
            sum(summary.elapsed_ticks for summary in hits) / len(hits) if hits else None
        ),
        "accepted_strikes": sum(summary.accepted_strikes for summary in summaries),
        "completed_misses": sum(summary.completed_misses for summary in summaries),
        "cooldown_violations": sum(summary.cooldown_violations for summary in summaries),
        "episodes_detail": [asdict(summary) | {"hit": summary.hit} for summary in summaries],
    }


def write_parquet(rollouts: list[Rollout], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as parquet

    rows = [row for rollout in rollouts for row in rollout.rows]
    if not rows:
        raise ValueError("cannot write an empty rollout dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    metadata = dict(table.schema.metadata or {})
    metadata[b"flybrain.dataset_schema_version"] = str(DATASET_SCHEMA_VERSION).encode()
    metadata[b"flybrain.observation_schema_version"] = str(
        flybrain_game.OBSERVATION_SCHEMA_VERSION
    ).encode()
    metadata[b"flybrain.action_schema_version"] = str(
        flybrain_game.ACTION_SCHEMA_VERSION
    ).encode()
    parquet.write_table(table.replace_schema_metadata(metadata), path, compression="zstd")


def write_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def write_replay_html(rollout: Rollout, path: Path) -> None:
    frames = [
        {
            "tick": row["tick"],
            "p": [row["player_x_units"], row["player_y_units"]],
            "l": [row["left_hand_x_units"], row["left_hand_y_units"]],
            "r": [row["right_hand_x_units"], row["right_hand_y_units"]],
            "lp": row["left_hand_phase"],
            "rp": row["right_hand_phase"],
            "lc": row["left_hand_active_contact"],
            "rc": row["right_hand_active_contact"],
            "target": [row["action_target_x_units"], row["action_target_y_units"]],
            "strike": row["action_strike"],
            "gap": row["contact_gap_units"],
            "status": row["post_round_status"],
        }
        for row in rollout.rows
    ]
    template = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>FlyBrain expert replay</title>
<style>body{margin:0;background:#18130f;color:#f8e7bc;font:16px system-ui;display:grid;place-items:center;min-height:100vh}main{width:min(960px,96vw)}canvas{width:100%;background:#d8b36d;border:2px solid #533b2d;border-radius:12px}div{display:flex;gap:12px;align-items:center;margin-top:8px}input{flex:1}</style>
<main><canvas width="960" height="540"></canvas><div><button>Pause</button><input type="range" min="0" value="0"><output></output></div></main>
<script>
const frames=__FRAMES__,scale=1024,canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d'),slider=document.querySelector('input'),out=document.querySelector('output'),button=document.querySelector('button');
slider.max=Math.max(0,frames.length-1);let playing=true,index=0,last=0;
function circle(point,r,fill,stroke='#4c2e25'){ctx.beginPath();ctx.arc(point[0]/scale,point[1]/scale,r,0,Math.PI*2);ctx.fillStyle=fill;ctx.fill();ctx.lineWidth=3;ctx.strokeStyle=stroke;ctx.stroke()}
function draw(){const f=frames[index];ctx.clearRect(0,0,960,540);ctx.fillStyle='#8b673f';ctx.fillRect(0,438,960,102);ctx.setLineDash([6,6]);ctx.strokeStyle='#6b382f';ctx.beginPath();ctx.moveTo(f.target[0]/scale-10,f.target[1]/scale);ctx.lineTo(f.target[0]/scale+10,f.target[1]/scale);ctx.moveTo(f.target[0]/scale,f.target[1]/scale-10);ctx.lineTo(f.target[0]/scale,f.target[1]/scale+10);ctx.stroke();ctx.setLineDash([]);circle(f.l,30,f.lc?'#ff5757':'#f4c59d');circle(f.r,30,f.rc?'#ff5757':'#f4c59d');circle(f.p,12,'#6ce6ff');out.value=`${__NAME__}  tick ${f.tick}  phases ${f.lp}/${f.rp}  gap ${(f.gap/scale).toFixed(1)} px  status ${f.status}`;slider.value=index}
function loop(time){if(playing&&time-last>1000/60){index=Math.min(index+1,frames.length-1);last=time;if(index===frames.length-1){playing=false;button.textContent='Play'}}draw();requestAnimationFrame(loop)}
button.onclick=()=>{playing=!playing;button.textContent=playing?'Pause':'Play'};slider.oninput=()=>{index=Number(slider.value);playing=false;button.textContent='Play';draw()};draw();requestAnimationFrame(loop);
</script></html>"""
    html = template.replace("__FRAMES__", json.dumps(frames)).replace(
        "__NAME__", json.dumps(rollout.summary.trajectory)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
