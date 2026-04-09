from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

EPS = 1e-10
max_oral_formants = 6


class GlottalSourceType(Enum):
    IMPULSIVE = "impulsive"
    NATURAL = "natural"
    NOISE = "noise"


glottal_source_type_enum_names = ["impulsive", "natural", "noise"]


@dataclass
class MainParms:
    sample_rate: int
    glottal_source_type: GlottalSourceType


@dataclass
class FrameParms:
    duration: float
    f0: float
    flutter_level: float
    open_phase_ratio: float
    breathiness_db: float
    tilt_db: float
    gain_db: float
    agc_rms_level: float
    nasal_formant_freq: float
    nasal_formant_bw: float
    oral_formant_freq: list[float] = field(default_factory=list)
    oral_formant_bw: list[float] = field(default_factory=list)
    cascade_enabled: bool = True
    cascade_voicing_db: float = 0.0
    cascade_aspiration_db: float = -99.0
    cascade_aspiration_mod: float = 0.0
    nasal_antiformant_freq: float = float("nan")
    nasal_antiformant_bw: float = float("nan")
    parallel_enabled: bool = False
    parallel_voicing_db: float = 0.0
    parallel_aspiration_db: float = -99.0
    parallel_aspiration_mod: float = 0.0
    frication_db: float = -99.0
    frication_mod: float = 0.0
    parallel_bypass_db: float = -99.0
    nasal_formant_db: float = float("nan")
    oral_formant_db: list[float] = field(default_factory=list)


@dataclass
class FrameState:
    breathiness_lin: float = 0.0
    gain_lin: float = 1.0
    cascade_voicing_lin: float = 0.0
    cascade_aspiration_lin: float = 0.0
    parallel_voicing_lin: float = 0.0
    parallel_aspiration_lin: float = 0.0
    frication_lin: float = 0.0
    parallel_bypass_lin: float = 0.0


@dataclass
class PeriodState:
    f0: float = 0.0
    period_length: int = 1
    open_phase_length: int = 0
    position_in_period: int = 0


def _poly_trim(poly: list[float], eps: float = EPS) -> list[float]:
    out = list(poly)
    while len(out) > 1 and abs(out[-1]) <= eps:
        out.pop()
    return out


def _poly_mul(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            out[i + j] += av * bv
    return _poly_trim(out)


def _poly_add(a: list[float], b: list[float]) -> list[float]:
    n = max(len(a), len(b))
    out = [0.0] * n
    for i in range(n):
        av = a[i] if i < len(a) else 0.0
        bv = b[i] if i < len(b) else 0.0
        out[i] = av + bv
    return _poly_trim(out)


def _multiply_fractions(
    left: tuple[list[float], list[float]],
    right: tuple[list[float], list[float]],
) -> tuple[list[float], list[float]]:
    return (_poly_mul(left[0], right[0]), _poly_mul(left[1], right[1]))


def _add_fractions(
    left: tuple[list[float], list[float]],
    right: tuple[list[float], list[float]],
) -> tuple[list[float], list[float]]:
    num = _poly_add(_poly_mul(left[0], right[1]), _poly_mul(right[0], left[1]))
    den = _poly_mul(left[1], right[1])
    return (_poly_trim(num), _poly_trim(den))


def _is_positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def get_white_noise() -> float:
    return random.random() * 2.0 - 1.0


class LpFilter1:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.a = 0.0
        self.b = 0.0
        self.y1 = 0.0
        self.passthrough = True
        self.muted = False

    def set(self, f: float, g: float, extra_gain: float = 1.0) -> None:
        if (
            f <= 0
            or f >= self.sample_rate / 2
            or g <= 0
            or g >= 1
            or not math.isfinite(f)
            or not math.isfinite(g)
            or not math.isfinite(extra_gain)
        ):
            raise ValueError("Invalid filter parameters.")
        w = 2 * math.pi * f / self.sample_rate
        q = (1 - g**2 * math.cos(w)) / (1 - g**2)
        self.b = q - math.sqrt(q**2 - 1)
        self.a = (1 - self.b) * extra_gain
        self.passthrough = False
        self.muted = False

    def set_passthrough(self) -> None:
        self.passthrough = True
        self.muted = False
        self.y1 = 0.0

    def set_mute(self) -> None:
        self.passthrough = False
        self.muted = True
        self.y1 = 0.0

    def get_transfer_function_coefficients(self) -> tuple[list[float], list[float]]:
        if self.passthrough:
            return ([1.0], [1.0])
        if self.muted:
            return ([0.0], [1.0])
        return ([self.a], [1.0, -self.b])

    def step(self, x: float) -> float:
        if self.passthrough:
            return x
        if self.muted:
            return 0.0
        y = self.a * x + self.b * self.y1
        self.y1 = y
        return y


class Resonator:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        self.r = 0.0
        self.passthrough = True
        self.muted = False

    def set(self, f: float, bw: float, dc_gain: float = 1.0) -> None:
        if (
            f < 0
            or f >= self.sample_rate / 2
            or bw <= 0
            or dc_gain <= 0
            or not math.isfinite(f)
            or not math.isfinite(bw)
            or not math.isfinite(dc_gain)
        ):
            raise ValueError("Invalid resonator parameters.")
        self.r = math.exp(-math.pi * bw / self.sample_rate)
        w = 2 * math.pi * f / self.sample_rate
        self.c = -(self.r**2)
        self.b = 2 * self.r * math.cos(w)
        self.a = (1 - self.b - self.c) * dc_gain
        self.passthrough = False
        self.muted = False

    def set_passthrough(self) -> None:
        self.passthrough = True
        self.muted = False
        self.y1 = 0.0
        self.y2 = 0.0

    def set_mute(self) -> None:
        self.passthrough = False
        self.muted = True
        self.y1 = 0.0
        self.y2 = 0.0

    def adjust_impulse_gain(self, new_a: float) -> None:
        self.a = new_a

    def adjust_peak_gain(self, peak_gain: float) -> None:
        if peak_gain <= 0 or not math.isfinite(peak_gain):
            raise ValueError("Invalid resonator peak gain.")
        self.a = peak_gain * (1 - self.r)

    def get_transfer_function_coefficients(self) -> tuple[list[float], list[float]]:
        if self.passthrough:
            return ([1.0], [1.0])
        if self.muted:
            return ([0.0], [1.0])
        return ([self.a], [1.0, -self.b, -self.c])

    def step(self, x: float) -> float:
        if self.passthrough:
            return x
        if self.muted:
            return 0.0
        y = self.a * x + self.b * self.y1 + self.c * self.y2
        self.y2 = self.y1
        self.y1 = y
        return y


class AntiResonator:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0
        self.x1 = 0.0
        self.x2 = 0.0
        self.passthrough = True
        self.muted = False

    def set(self, f: float, bw: float) -> None:
        if (
            f <= 0
            or f >= self.sample_rate / 2
            or bw <= 0
            or not math.isfinite(f)
            or not math.isfinite(bw)
        ):
            raise ValueError("Invalid anti-resonator parameters.")
        r = math.exp(-math.pi * bw / self.sample_rate)
        w = 2 * math.pi * f / self.sample_rate
        c0 = -(r * r)
        b0 = 2 * r * math.cos(w)
        a0 = 1 - b0 - c0
        if a0 == 0:
            self.a = 0.0
            self.b = 0.0
            self.c = 0.0
            return
        self.a = 1 / a0
        self.b = -b0 / a0
        self.c = -c0 / a0
        self.passthrough = False
        self.muted = False

    def set_passthrough(self) -> None:
        self.passthrough = True
        self.muted = False
        self.x1 = 0.0
        self.x2 = 0.0

    def set_mute(self) -> None:
        self.passthrough = False
        self.muted = True
        self.x1 = 0.0
        self.x2 = 0.0

    def get_transfer_function_coefficients(self) -> tuple[list[float], list[float]]:
        if self.passthrough:
            return ([1.0], [1.0])
        if self.muted:
            return ([0.0], [1.0])
        return ([self.a, self.b, self.c], [1.0])

    def step(self, x: float) -> float:
        if self.passthrough:
            return x
        if self.muted:
            return 0.0
        y = self.a * x + self.b * self.x1 + self.c * self.x2
        self.x2 = self.x1
        self.x1 = x
        return y


class DifferencingFilter:
    def __init__(self) -> None:
        self.x1 = 0.0

    def get_transfer_function_coefficients(self) -> tuple[list[float], list[float]]:
        return ([1.0, -1.0], [1.0])

    def step(self, x: float) -> float:
        y = x - self.x1
        self.x1 = x
        return y


class LpNoiseSource:
    def __init__(self, sample_rate: int) -> None:
        old_b = 0.75
        old_sample_rate = 10000.0
        f = 1000.0
        g = (1 - old_b) / math.sqrt(
            1 - 2 * old_b * math.cos(2 * math.pi * f / old_sample_rate) + old_b**2
        )
        extra_gain = 2.5 * (sample_rate / 10000.0) ** 0.33
        self.lp_filter = LpFilter1(sample_rate)
        self.lp_filter.set(f, g, extra_gain)

    def get_next(self) -> float:
        return self.lp_filter.step(get_white_noise())


class ImpulsiveGlottalSource:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.resonator: Resonator | None = None
        self.position_in_period = 0

    def start_period(self, open_phase_length: int) -> None:
        if not open_phase_length:
            self.resonator = None
            return
        if self.resonator is None:
            self.resonator = Resonator(self.sample_rate)
        bw = self.sample_rate / open_phase_length
        self.resonator.set(0.0, bw)
        self.resonator.adjust_impulse_gain(1.0)
        self.position_in_period = 0

    def get_next(self) -> float:
        if self.resonator is None:
            return 0.0
        pulse = 1.0 if self.position_in_period == 1 else -1.0 if self.position_in_period == 2 else 0.0
        self.position_in_period += 1
        return self.resonator.step(pulse)


class NaturalGlottalSource:
    def __init__(self) -> None:
        self.x = 0.0
        self.a = 0.0
        self.b = 0.0
        self.open_phase_length = 0
        self.position_in_period = 0

    def start_period(self, open_phase_length: int) -> None:
        self.open_phase_length = open_phase_length
        self.x = 0.0
        self.position_in_period = 0
        if open_phase_length <= 0:
            self.a = 0.0
            self.b = 0.0
            return
        amplification = 5.0
        self.b = -amplification / (open_phase_length**2)
        self.a = -self.b * open_phase_length / 3.0

    def get_next(self) -> float:
        if self.position_in_period >= self.open_phase_length:
            self.position_in_period += 1
            self.x = 0.0
            return 0.0
        self.position_in_period += 1
        self.a += self.b
        self.x += self.a
        return self.x


def perform_frequency_modulation(f0: float, flutter_level: float, time_s: float) -> float:
    if flutter_level <= 0:
        return f0
    w = 2 * math.pi * time_s
    a = math.sin(12.7 * w) + math.sin(7.1 * w) + math.sin(4.7 * w)
    return f0 * (1 + a * flutter_level / 50.0)


def db_to_lin(db: float) -> float:
    if math.isnan(db) or db <= -99:
        return 0.0
    return 10 ** (db / 20.0)


def set_tilt_filter(tilt_filter: LpFilter1, tilt_db: float) -> None:
    if not tilt_db:
        tilt_filter.set_passthrough()
    else:
        tilt_filter.set(3000.0, db_to_lin(-tilt_db))


def set_nasal_formant_casc(nasal_formant_casc: Resonator, f_parms: FrameParms) -> None:
    if _is_positive_finite(f_parms.nasal_formant_freq) and _is_positive_finite(f_parms.nasal_formant_bw):
        nasal_formant_casc.set(f_parms.nasal_formant_freq, f_parms.nasal_formant_bw)
    else:
        nasal_formant_casc.set_passthrough()


def set_nasal_antiformant_casc(nasal_antiformant_casc: AntiResonator, f_parms: FrameParms) -> None:
    if _is_positive_finite(f_parms.nasal_antiformant_freq) and _is_positive_finite(f_parms.nasal_antiformant_bw):
        nasal_antiformant_casc.set(f_parms.nasal_antiformant_freq, f_parms.nasal_antiformant_bw)
    else:
        nasal_antiformant_casc.set_passthrough()


def set_oral_formant_casc(oral_formant_casc: Resonator, f_parms: FrameParms, i: int) -> None:
    f = f_parms.oral_formant_freq[i] if i < len(f_parms.oral_formant_freq) else float("nan")
    bw = f_parms.oral_formant_bw[i] if i < len(f_parms.oral_formant_bw) else float("nan")
    if _is_positive_finite(f) and _is_positive_finite(bw):
        oral_formant_casc.set(f, bw)
    else:
        oral_formant_casc.set_passthrough()


def set_nasal_formant_par(nasal_formant_par: Resonator, f_parms: FrameParms) -> None:
    nasal_gain = db_to_lin(f_parms.nasal_formant_db)
    if (
        _is_positive_finite(f_parms.nasal_formant_freq)
        and _is_positive_finite(f_parms.nasal_formant_bw)
        and nasal_gain > 0
    ):
        nasal_formant_par.set(f_parms.nasal_formant_freq, f_parms.nasal_formant_bw)
        nasal_formant_par.adjust_peak_gain(nasal_gain)
    else:
        nasal_formant_par.set_mute()


def set_oral_formant_par(
    oral_formant_par: Resonator,
    m_parms: MainParms,
    f_parms: FrameParms,
    i: int,
) -> None:
    formant = i + 1
    f = f_parms.oral_formant_freq[i] if i < len(f_parms.oral_formant_freq) else float("nan")
    bw = f_parms.oral_formant_bw[i] if i < len(f_parms.oral_formant_bw) else float("nan")
    db = f_parms.oral_formant_db[i] if i < len(f_parms.oral_formant_db) else float("nan")
    peak_gain = db_to_lin(db)
    if _is_positive_finite(f) and _is_positive_finite(bw) and peak_gain > 0:
        oral_formant_par.set(f, bw)
        w = 2 * math.pi * f / m_parms.sample_rate
        diff_gain = math.sqrt(2 - 2 * math.cos(w))
        filter_gain = peak_gain / diff_gain if formant >= 2 else peak_gain
        oral_formant_par.adjust_peak_gain(filter_gain)
    else:
        oral_formant_par.set_mute()


def compute_rms(buf: list[float]) -> float:
    if not buf:
        return 0.0
    return math.sqrt(sum(x * x for x in buf) / len(buf))


def find_max_abs_value(buf: list[float]) -> float:
    return max((abs(x) for x in buf), default=0.0)


def adjust_signal_gain(buf: list[float], target_rms: float) -> None:
    if not buf:
        return
    rms = compute_rms(buf)
    if rms == 0:
        return
    r = target_rms / rms
    max_abs = find_max_abs_value(buf)
    if max_abs > 0 and r * max_abs >= 1:
        r = 0.99 / max_abs
    for i, value in enumerate(buf):
        buf[i] = value * r


class Generator:
    def __init__(self, m_parms: MainParms) -> None:
        self.m_parms = m_parms
        self.f_parms: FrameParms | None = None
        self.new_f_parms: FrameParms | None = None
        self.f_state = FrameState()
        self.p_state: PeriodState | None = None
        self.abs_position = 0
        self.tilt_filter = LpFilter1(m_parms.sample_rate)
        self.flutter_time_offset = random.random() * 1000.0
        self.output_lp_filter = Resonator(m_parms.sample_rate)
        self.output_lp_filter.set(0.0, m_parms.sample_rate / 2.0)

        self.impulsive_g_source: ImpulsiveGlottalSource | None = None
        self.natural_g_source: NaturalGlottalSource | None = None
        self.glottal_source: Callable[[], float]
        self._init_glottal_source()

        self.aspiration_source_casc = LpNoiseSource(m_parms.sample_rate)
        self.aspiration_source_par = LpNoiseSource(m_parms.sample_rate)
        self.frication_source_par = LpNoiseSource(m_parms.sample_rate)

        self.nasal_formant_casc = Resonator(m_parms.sample_rate)
        self.nasal_antiformant_casc = AntiResonator(m_parms.sample_rate)
        self.oral_formant_casc = [Resonator(m_parms.sample_rate) for _ in range(max_oral_formants)]

        self.nasal_formant_par = Resonator(m_parms.sample_rate)
        self.oral_formant_par = [Resonator(m_parms.sample_rate) for _ in range(max_oral_formants)]
        self.differencing_filter_par = DifferencingFilter()

    def generate_frame(self, f_parms: FrameParms, out_len: int) -> list[float]:
        self.new_f_parms = f_parms
        out_buf: list[float] = [0.0] * out_len
        for out_pos in range(out_len):
            if self.p_state is None or self.p_state.position_in_period >= self.p_state.period_length:
                self._start_new_period()
            out_buf[out_pos] = self._compute_next_output_signal_sample()
            assert self.p_state is not None
            self.p_state.position_in_period += 1
            self.abs_position += 1
        if math.isnan(f_parms.gain_db):
            adjust_signal_gain(out_buf, f_parms.agc_rms_level)
        return out_buf

    def _compute_next_output_signal_sample(self) -> float:
        assert self.f_parms is not None
        assert self.p_state is not None
        voice = self.glottal_source()
        voice = self.tilt_filter.step(voice)
        if self.p_state.position_in_period < self.p_state.open_phase_length:
            voice += get_white_noise() * self.f_state.breathiness_lin
        cascade_out = self._compute_cascade_branch(voice) if self.f_parms.cascade_enabled else 0.0
        parallel_out = self._compute_parallel_branch(voice) if self.f_parms.parallel_enabled else 0.0
        out = cascade_out + parallel_out
        out = self.output_lp_filter.step(out)
        out *= self.f_state.gain_lin
        return out

    def _compute_cascade_branch(self, voice: float) -> float:
        assert self.f_parms is not None
        assert self.p_state is not None
        cascade_voice = voice * self.f_state.cascade_voicing_lin
        current_asp_mod = (
            self.f_parms.cascade_aspiration_mod
            if self.p_state.position_in_period >= self.p_state.period_length / 2
            else 0.0
        )
        aspiration = (
            self.aspiration_source_casc.get_next()
            * self.f_state.cascade_aspiration_lin
            * (1 - current_asp_mod)
        )
        v = cascade_voice + aspiration
        v = self.nasal_antiformant_casc.step(v)
        v = self.nasal_formant_casc.step(v)
        for resonator in self.oral_formant_casc:
            v = resonator.step(v)
        return v

    def _compute_parallel_branch(self, voice: float) -> float:
        assert self.f_parms is not None
        assert self.p_state is not None
        parallel_voice = voice * self.f_state.parallel_voicing_lin
        current_asp_mod = (
            self.f_parms.parallel_aspiration_mod
            if self.p_state.position_in_period >= self.p_state.period_length / 2
            else 0.0
        )
        aspiration = (
            self.aspiration_source_par.get_next()
            * self.f_state.parallel_aspiration_lin
            * (1 - current_asp_mod)
        )
        source = parallel_voice + aspiration
        source_difference = self.differencing_filter_par.step(source)
        current_fric_mod = (
            self.f_parms.frication_mod if self.p_state.position_in_period >= self.p_state.period_length / 2 else 0.0
        )
        frication_noise = self.frication_source_par.get_next() * self.f_state.frication_lin * (1 - current_fric_mod)
        source2 = source_difference + frication_noise
        v = 0.0
        v += self.nasal_formant_par.step(source)
        v += self.oral_formant_par[0].step(source)
        for i in range(1, max_oral_formants):
            alternating_sign = 1.0 if i % 2 == 0 else -1.0
            v += alternating_sign * self.oral_formant_par[i].step(source2)
        v += self.f_state.parallel_bypass_lin * source2
        return v

    def _start_new_period(self) -> None:
        if self.new_f_parms is not None:
            self.f_parms = self.new_f_parms
            self.new_f_parms = None
            self._start_using_new_frame_parameters()
        if self.p_state is None:
            self.p_state = PeriodState()
        assert self.f_parms is not None
        flutter_time = self.abs_position / self.m_parms.sample_rate + self.flutter_time_offset
        self.p_state.f0 = perform_frequency_modulation(self.f_parms.f0, self.f_parms.flutter_level, flutter_time)
        self.p_state.period_length = (
            round(self.m_parms.sample_rate / self.p_state.f0) if self.p_state.f0 > 0 else 1
        )
        self.p_state.open_phase_length = (
            round(self.p_state.period_length * self.f_parms.open_phase_ratio)
            if self.p_state.period_length > 1
            else 0
        )
        self.p_state.position_in_period = 0
        self._start_glottal_source_period()

    def _start_using_new_frame_parameters(self) -> None:
        assert self.f_parms is not None
        self.f_state.breathiness_lin = db_to_lin(self.f_parms.breathiness_db)
        self.f_state.gain_lin = db_to_lin(self.f_parms.gain_db if not math.isnan(self.f_parms.gain_db) else 0.0)
        set_tilt_filter(self.tilt_filter, self.f_parms.tilt_db)

        self.f_state.cascade_voicing_lin = db_to_lin(self.f_parms.cascade_voicing_db)
        self.f_state.cascade_aspiration_lin = db_to_lin(self.f_parms.cascade_aspiration_db)
        set_nasal_formant_casc(self.nasal_formant_casc, self.f_parms)
        set_nasal_antiformant_casc(self.nasal_antiformant_casc, self.f_parms)
        for i in range(max_oral_formants):
            set_oral_formant_casc(self.oral_formant_casc[i], self.f_parms, i)

        self.f_state.parallel_voicing_lin = db_to_lin(self.f_parms.parallel_voicing_db)
        self.f_state.parallel_aspiration_lin = db_to_lin(self.f_parms.parallel_aspiration_db)
        self.f_state.frication_lin = db_to_lin(self.f_parms.frication_db)
        self.f_state.parallel_bypass_lin = db_to_lin(self.f_parms.parallel_bypass_db)
        set_nasal_formant_par(self.nasal_formant_par, self.f_parms)
        for i in range(max_oral_formants):
            set_oral_formant_par(self.oral_formant_par[i], self.m_parms, self.f_parms, i)

    def _init_glottal_source(self) -> None:
        if self.m_parms.glottal_source_type == GlottalSourceType.IMPULSIVE:
            self.impulsive_g_source = ImpulsiveGlottalSource(self.m_parms.sample_rate)
            self.glottal_source = self.impulsive_g_source.get_next
        elif self.m_parms.glottal_source_type == GlottalSourceType.NATURAL:
            self.natural_g_source = NaturalGlottalSource()
            self.glottal_source = self.natural_g_source.get_next
        elif self.m_parms.glottal_source_type == GlottalSourceType.NOISE:
            self.glottal_source = get_white_noise
        else:
            raise ValueError("Undefined glottal source type.")

    def _start_glottal_source_period(self) -> None:
        assert self.p_state is not None
        if self.m_parms.glottal_source_type == GlottalSourceType.IMPULSIVE:
            assert self.impulsive_g_source is not None
            self.impulsive_g_source.start_period(self.p_state.open_phase_length)
        elif self.m_parms.glottal_source_type == GlottalSourceType.NATURAL:
            assert self.natural_g_source is not None
            self.natural_g_source.start_period(self.p_state.open_phase_length)


def generate_sound(m_parms: MainParms, f_parms_a: list[FrameParms]) -> list[float]:
    generator = Generator(m_parms)
    out_buf: list[float] = []
    for f_parms in f_parms_a:
        frame_len = round(f_parms.duration * m_parms.sample_rate)
        out_buf.extend(generator.generate_frame(f_parms, frame_len))
    return out_buf


def get_vocal_tract_transfer_function_coefficients(
    m_parms: MainParms,
    f_parms: FrameParms,
) -> tuple[list[float], list[float]]:
    voice: tuple[list[float], list[float]] = ([1.0], [1.0])

    tilt_filter = LpFilter1(m_parms.sample_rate)
    set_tilt_filter(tilt_filter, f_parms.tilt_db)
    voice = _multiply_fractions(voice, tilt_filter.get_transfer_function_coefficients())

    cascade_trans = (
        _get_cascade_branch_transfer_function_coefficients(m_parms, f_parms)
        if f_parms.cascade_enabled
        else ([0.0], [1.0])
    )
    parallel_trans = (
        _get_parallel_branch_transfer_function_coefficients(m_parms, f_parms)
        if f_parms.parallel_enabled
        else ([0.0], [1.0])
    )
    branches_trans = _add_fractions(cascade_trans, parallel_trans)
    out = _multiply_fractions(voice, branches_trans)

    output_lp_filter = Resonator(m_parms.sample_rate)
    output_lp_filter.set(0.0, m_parms.sample_rate / 2.0)
    out = _multiply_fractions(out, output_lp_filter.get_transfer_function_coefficients())

    gain_lin = db_to_lin(f_parms.gain_db if not math.isnan(f_parms.gain_db) else 0.0)
    out = _multiply_fractions(out, ([gain_lin], [1.0]))
    return out


def _get_cascade_branch_transfer_function_coefficients(
    m_parms: MainParms,
    f_parms: FrameParms,
) -> tuple[list[float], list[float]]:
    v: tuple[list[float], list[float]] = ([db_to_lin(f_parms.cascade_voicing_db)], [1.0])

    nasal_antiformant_casc = AntiResonator(m_parms.sample_rate)
    set_nasal_antiformant_casc(nasal_antiformant_casc, f_parms)
    v = _multiply_fractions(v, nasal_antiformant_casc.get_transfer_function_coefficients())

    nasal_formant_casc = Resonator(m_parms.sample_rate)
    set_nasal_formant_casc(nasal_formant_casc, f_parms)
    v = _multiply_fractions(v, nasal_formant_casc.get_transfer_function_coefficients())

    for i in range(max_oral_formants):
        oral_formant_casc = Resonator(m_parms.sample_rate)
        set_oral_formant_casc(oral_formant_casc, f_parms, i)
        v = _multiply_fractions(v, oral_formant_casc.get_transfer_function_coefficients())
    return v


def _get_parallel_branch_transfer_function_coefficients(
    m_parms: MainParms,
    f_parms: FrameParms,
) -> tuple[list[float], list[float]]:
    source: tuple[list[float], list[float]] = ([db_to_lin(f_parms.parallel_voicing_db)], [1.0])

    differencing_filter_par = DifferencingFilter()
    source2 = _multiply_fractions(source, differencing_filter_par.get_transfer_function_coefficients())

    v: tuple[list[float], list[float]] = ([0.0], [1.0])

    nasal_formant_par = Resonator(m_parms.sample_rate)
    set_nasal_formant_par(nasal_formant_par, f_parms)
    nasal_out = _multiply_fractions(source, nasal_formant_par.get_transfer_function_coefficients())
    v = _add_fractions(v, nasal_out)

    for i in range(max_oral_formants):
        oral_formant_par = Resonator(m_parms.sample_rate)
        set_oral_formant_par(oral_formant_par, m_parms, f_parms, i)
        formant_in = source if i == 0 else source2
        formant_out = _multiply_fractions(formant_in, oral_formant_par.get_transfer_function_coefficients())
        alternating_sign = 1.0 if i % 2 == 0 else -1.0
        v2 = _multiply_fractions(formant_out, ([alternating_sign], [1.0]))
        v = _add_fractions(v, v2)

    parallel_bypass = _multiply_fractions(source2, ([db_to_lin(f_parms.parallel_bypass_db)], [1.0]))
    v = _add_fractions(v, parallel_bypass)
    return v
