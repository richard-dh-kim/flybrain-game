"""Interception expert used to label conventional baseline training data."""

from __future__ import annotations

from dataclasses import dataclass

import flybrain_game

from .schema import ActionV1, ObservationV1, synchronized_strike_is_ready


@dataclass(frozen=True, slots=True)
class InterceptExpertV1:
    first_strike_tick: int = 45
    velocity_lead_scale_numerator: int = 1
    velocity_lead_scale_denominator: int = 1

    @property
    def contact_lead_ticks(self) -> int:
        """Ticks from a command until the active part of the strike opens."""

        return flybrain_game.HAND_WIND_UP_TICKS + (
            flybrain_game.HAND_STRIKE_TICKS - flybrain_game.HAND_ACTIVE_CONTACT_TICKS
        )

    def decide(self, observation: ObservationV1) -> ActionV1:
        lead_numerator = self.contact_lead_ticks * self.velocity_lead_scale_numerator
        lead_denominator = max(1, self.velocity_lead_scale_denominator)
        target_x = observation.player_position[0] + (
            observation.player_velocity[0] * lead_numerator // lead_denominator
        )
        target_y = observation.player_position[1] + (
            observation.player_velocity[1] * lead_numerator // lead_denominator
        )
        target = (
            min(
                flybrain_game.FLIGHT_MAX_X_UNITS,
                max(flybrain_game.FLIGHT_MIN_X_UNITS, target_x),
            ),
            min(
                flybrain_game.FLIGHT_MAX_Y_UNITS,
                max(flybrain_game.FLIGHT_MIN_Y_UNITS, target_y),
            ),
        )
        strike = observation.tick >= self.first_strike_tick and synchronized_strike_is_ready(
            observation
        )
        return ActionV1(target=target, strike=strike)
