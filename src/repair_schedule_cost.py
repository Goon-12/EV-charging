
import numpy as np


def repair_schedule_cost(
    schedule,
    scenario,
    prices,
    step_hours=0.25,
    n_slots=144,
    max_chargers=20,
    max_grid_kw=125.0,
):
    """
    Rebuild a feasible schedule, considering low-price slots first.

    Hard constraints:
    - EV arrival/departure windows
    - Individual EV power limits
    - Maximum number of simultaneous chargers
    - Aggregate grid capacity

    Candidate power preferences help break ties between EVs.
    """
    candidate = np.asarray(schedule, dtype=float)
    prices = np.asarray(prices, dtype=float)

    n_ev = len(scenario)

    if candidate.shape != (n_ev, n_slots):
        raise ValueError("Candidate schedule has an unexpected shape.")

    if prices.shape != (n_slots,):
        raise ValueError("Tariff must have one price per time slot.")

    if (
        not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(prices))
        or np.any(prices < 0)
    ):
        raise ValueError("Schedule and tariff must be finite and valid.")

    arrival = scenario["arrival_slot"].to_numpy(dtype=int)
    departure = scenario["departure_slot"].to_numpy(dtype=int)
    required = scenario["required_energy_kwh"].to_numpy(dtype=float)
    max_power = scenario["max_power_kw"].to_numpy(dtype=float)

    if np.any(required < 0) or np.any(max_power <= 0):
        raise ValueError("Invalid energy requirements or power limits.")

    starts = np.clip(arrival, 0, n_slots)
    ends = np.clip(departure, 0, n_slots)

    # Ignore inherited power outside valid windows or above power limits.
    candidate = np.clip(candidate, 0.0, max_power[:, None])

    for i in range(n_ev):
        candidate[i, :starts[i]] = 0.0
        candidate[i, ends[i]:] = 0.0

    # Prefer lower-price slots. For equal prices, use earlier slots first.
    slot_order = sorted(range(n_slots), key=lambda t: (prices[t], t))

    repaired = np.zeros((n_ev, n_slots), dtype=float)
    delivered = np.zeros(n_ev, dtype=float)

    for t in slot_order:
        eligible = [
            i for i in range(n_ev)
            if starts[i] <= t < ends[i]
            and required[i] - delivered[i] > 1e-9
        ]

        # Earlier departures first; then prefer the candidate's
        # requested charging slots and larger requested power.
        eligible.sort(
            key=lambda i: (
                ends[i],
                -int(candidate[i, t] > 1e-9)
                -candidate[i, t],
                -(required[i] - delivered[i]),
                i,
            )
        )

        grid_remaining = max_grid_kw
        active_count = 0

        for i in eligible:
            if active_count >= max_chargers or grid_remaining <= 1e-9:
                break

            remaining_energy = required[i] - delivered[i]

            power = min(
                max_power[i],
                remaining_energy / step_hours,
                grid_remaining,
            )

            if power > 1e-9:
                repaired[i, t] = power
                delivered[i] += power * step_hours
                grid_remaining -= power
                active_count += 1

    return repaired
