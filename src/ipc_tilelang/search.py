from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .config import Fidelity, Measurement, ProblemKey, SearchConfig, SearchResult, SQLiteCache
from .model import FactorizedGaussianSurrogate, config_vector


@dataclass
class Observation:
    cfg: Any
    measurement: Measurement
    round_id: int


@dataclass
class Region:
    prefix: tuple[tuple[str, Any], ...]
    indices: tuple[int, ...]
    depth: int
    parent: "Region | None" = None
    children: tuple["Region", ...] = ()
    prune_streak: int = 0
    pruned: bool = False
    split: bool = False

    @property
    def key(self) -> str:
        return "/".join(f"{k}={v}" for k, v in self.prefix) or "ROOT"


class SemanticSearchTree:
    """Semantic partition of the complete legal finite configuration space."""

    LEVELS = (
        "block_m", "block_n", "block_k", "threads",
        "num_stages", "swizzle", "swizzle_panel", "shared_swizzle",
    )

    def __init__(self, configs: Sequence[Any]):
        self.configs = list(configs)
        self.root = Region((), tuple(range(len(self.configs))), 0)
        self.nodes: list[Region] = [self.root]

    @staticmethod
    def _value(cfg: Any, field: str) -> Any:
        return getattr(cfg, field)

    def split(self, region: Region) -> tuple[Region, ...]:
        if region.pruned or region.split or region.depth >= len(self.LEVELS) or len(region.indices) <= 1:
            return ()
        field = self.LEVELS[region.depth]
        buckets: dict[Any, list[int]] = defaultdict(list)
        for idx in region.indices:
            buckets[self._value(self.configs[idx], field)].append(idx)
        children: list[Region] = []
        for value in sorted(buckets, key=lambda x: str(x)):
            child = Region(
                prefix=region.prefix + ((field, value),),
                indices=tuple(buckets[value]),
                depth=region.depth + 1,
                parent=region,
            )
            children.append(child)
            self.nodes.append(child)
        region.children = tuple(children)
        region.split = True
        return region.children

    def active_leaves(self) -> list[Region]:
        return [
            n for n in self.nodes
            if not n.pruned and (not n.split or not n.children)
        ]

    def protected_path(self, cfg: Any | None) -> tuple[Region, ...]:
        if cfg is None:
            return ()
        return tuple(
            n for n in self.nodes
            if any(self.configs[i] == cfg for i in n.indices)
        )

    def mark_pruned(self, region: Region) -> None:
        if region.pruned:
            return
        region.pruned = True
        # Children are already logically included in the parent; marking them
        # too simplifies active-leaf accounting without changing the partition.
        stack = list(region.children)
        while stack:
            child = stack.pop()
            child.pruned = True
            stack.extend(child.children)

    def pruned_configs(self) -> set[Any]:
        out: set[Any] = set()
        for node in self.nodes:
            if node.pruned:
                out.update(self.configs[i] for i in node.indices)
        return out


def _cfg_key(cfg: Any) -> str:
    return json.dumps(cfg.__dict__, sort_keys=True, separators=(",", ":"))


def _distance(a: Any, b: Any) -> float:
    vals = (
        math.log2(a.block_m), math.log2(a.block_n), math.log2(a.block_k),
        a.threads / 32.0, float(a.num_stages), float(bool(a.swizzle)),
        float(bool(a.shared_swizzle)),
    )
    vals_b = (
        math.log2(b.block_m), math.log2(b.block_n), math.log2(b.block_k),
        b.threads / 32.0, float(b.num_stages), float(bool(b.swizzle)),
        float(bool(b.shared_swizzle)),
    )
    weights = (1.6, 1.6, 1.3, 0.9, 0.65, 0.35, 0.35)
    return math.sqrt(sum(w * (x - y) ** 2 for x, y, w in zip(vals, vals_b, weights)))


class AdaptiveFiniteTuner:
    """Unlimited finite-space autotuner with model-guided semantic branch pruning.

    The stopping rule is intentionally not a numeric evaluation budget. A search
    continues until every legal configuration is either measured/remembered or
    belongs to a region whose conservative model lower bound is worse than the
    incumbent. Thus a large space can finish after few launches, while an
    adversarial jagged space naturally falls back toward exhaustive evaluation.
    """

    ENGINE = "online_racing_v12_fast_knn_ei_batch_racing"

    def __init__(self, cache: SQLiteCache, search: SearchConfig):
        self.cache = cache
        self.search = search

    def _seed_order(self, configs: Sequence[Any], preferred: Sequence[Any]) -> list[Any]:
        """Deterministic startup design that deliberately probes the micro-tile basin.

        The startup points do not depend on persistent measurements. They are fixed
        hardware/workload priors plus geometric extremes, so a brand-new run can
        immediately build a useful model instead of waiting for a stale cache.
        """
        by_key = {_cfg_key(c): c for c in configs}
        seeds: list[Any] = []

        def add(cfg: Any) -> None:
            if _cfg_key(cfg) in by_key and cfg not in seeds:
                seeds.append(by_key[_cfg_key(cfg)])

        for cfg in preferred:
            add(cfg)

        # SM86/iPC micro-tile priors: small MxN tiles, 64/128 threads,
        # BK 16/32 and 2/3 stages are deliberately represented up front.
        anchor_values = (
            (16, 32, 16, 64, 3, False),
            (16, 32, 16, 64, 2, False),
            (16, 32, 16, 128, 3, False),
            (16, 32, 32, 64, 2, True),
            (16, 32, 32, 128, 2, True),
            (32, 32, 16, 64, 3, False),
            (32, 32, 16, 128, 2, True),
            (16, 64, 16, 64, 3, False),
        )
        for bm, bn, bk, th, st, sw in anchor_values:
            for c in configs:
                if (c.block_m, c.block_n, c.block_k, c.threads, c.num_stages, bool(c.swizzle)) == (bm, bn, bk, th, st, sw):
                    add(c)
                    break

        bm_vals = sorted({c.block_m for c in configs})
        bn_vals = sorted({c.block_n for c in configs})
        bk_vals = sorted({c.block_k for c in configs})
        th_vals = sorted({c.threads for c in configs})
        if bm_vals and bn_vals:
            for bm in (bm_vals[0], bm_vals[-1]):
                for bn in (bn_vals[0], bn_vals[-1]):
                    cand = min(
                        configs,
                        key=lambda c, bm=bm, bn=bn: (abs(c.block_m-bm), abs(c.block_n-bn), abs(c.block_k-bk_vals[0]), abs(c.threads-128), int(c.swizzle)),
                    )
                    add(cand)
        # Explicitly cover all thread counts before model-based racing begins.
        for th in th_vals:
            remaining = [c for c in configs if c not in seeds]
            if not remaining:
                break
            cand = min(remaining, key=lambda c, th=th: (abs(c.threads-th), abs(c.block_m-16), abs(c.block_n-32), abs(c.block_k-16), c.num_stages, int(c.swizzle)))
            add(cand)

        remaining = [c for c in configs if c not in seeds]
        while len(seeds) < min(self.search.seed_evals, len(configs)) and remaining:
            candidate = max(
                remaining,
                key=lambda c: (min((_distance(c, q) for q in seeds), default=3.0), _cfg_key(c)),
            )
            add(candidate)
            remaining.remove(candidate)
        return seeds[: self.search.seed_evals]

    @staticmethod
    def _neighbors(cfg: Any, configs: Sequence[Any]) -> set[Any]:
        """Compact Hamming-1/2 neighborhood using *adjacent* axis values only.

        v11's neighborhood allowed jumps of up to 4x on a tile axis. That produced
        dozens of points which were not local in the discrete performance landscape.
        v12 treats predecessor/successor values as the local racing surface and uses
        pairwise recombinations only among those adjacent values.
        """
        key_to_cfg = {_cfg_key(c): c for c in configs}
        axes = (
            ("block_m", sorted({c.block_m for c in configs})),
            ("block_n", sorted({c.block_n for c in configs})),
            ("block_k", sorted({c.block_k for c in configs})),
            ("threads", sorted({c.threads for c in configs})),
            ("num_stages", sorted({c.num_stages for c in configs})),
            ("swizzle", sorted({bool(c.swizzle) for c in configs})),
        )
        out: set[Any] = set()

        def make(values: dict[str, Any]) -> None:
            probe = dict(cfg.__dict__)
            probe.update(values)
            try:
                candidate = type(cfg)(**probe)
            except Exception:
                return
            canonical = key_to_cfg.get(_cfg_key(candidate))
            if canonical is not None and canonical != cfg:
                out.add(canonical)

        adjacent: dict[str, list[Any]] = {}
        for field, values in axes:
            cur = getattr(cfg, field)
            if cur not in values:
                continue
            i = values.index(cur)
            vals: list[Any] = []
            if i > 0:
                vals.append(values[i - 1])
            if i + 1 < len(values):
                vals.append(values[i + 1])
            adjacent[field] = vals
            for value in vals:
                make({field: value})

        pair_axes = (
            ("block_m", "block_n"),
            ("block_m", "block_k"),
            ("block_n", "block_k"),
            ("block_m", "threads"),
            ("block_n", "threads"),
            ("threads", "num_stages"),
            ("block_k", "num_stages"),
        )
        for a, b in pair_axes:
            for va in adjacent.get(a, ()):
                for vb in adjacent.get(b, ()):
                    make({a: va, b: vb})
        return out

    @staticmethod
    def _global_scouts(cfgs: Sequence[Any], observations: Sequence[Observation], limit: int = 24) -> list[Any]:
        """Deterministic farthest-point scouts for continued global coverage."""
        if not cfgs:
            return []
        obs = [o.cfg for o in observations]
        remaining = list(cfgs)
        chosen: list[Any] = []
        while remaining and len(chosen) < limit:
            ref = obs + chosen
            pick = max(
                remaining,
                key=lambda c: (min((_distance(c, x) for x in ref), default=3.0), _cfg_key(c)),
            )
            chosen.append(pick)
            remaining.remove(pick)
        return chosen

    def _fit_model(self, observations: Sequence[Observation], *, problem: ProblemKey):
        valid = [
            o for o in observations
            if o.measurement.successful and o.measurement.latency_ms is not None
        ]
        if len(valid) < 2:
            return None
        X = np.vstack([
            config_vector(o.cfg, M=problem.M, K=problem.K, B=problem.B, kind=problem.kind)
            for o in valid
        ])
        y = np.log(np.asarray([max(o.measurement.latency_ms, 1e-6) for o in valid], dtype=np.float64))
        model = FactorizedGaussianSurrogate(noise_floor=self.search.robust_noise_floor)
        model.fit(X, y)
        return model

    def _inflated_predictions(
        self,
        candidates: Sequence[Any],
        observations: Sequence[Observation],
        model: FactorizedGaussianSurrogate,
        *,
        problem: ProblemKey,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.vstack([
            config_vector(c, M=problem.M, K=problem.K, B=problem.B, kind=problem.kind)
            for c in candidates
        ])
        mean, std, lcb = model.predict_latency(X, beta=self.search.beta)
        return mean, std, lcb

    def _candidate_scores(
        self,
        candidates: Sequence[Any],
        observations: Sequence[Observation],
        model: FactorizedGaussianSurrogate | None,
        *,
        problem: ProblemKey,
        incumbent_cfg: Any | None = None,
        incumbent_ms: float = float("inf"),
    ) -> dict[Any, tuple[float, float, float, float, float, float]]:
        """Return mean/std/LCB/EI/racing_score/distance for online selection.

        EI is computed in log-space, matching the surrogate target and keeping
        numerical behaviour sane for sub-0.1ms kernels.
        """
        if not candidates:
            return {}
        obs_cfgs = [o.cfg for o in observations]
        if model is None:
            return {
                c: (float("inf"), float("inf"), float("-inf"), 0.0, 0.0,
                    min((_distance(c, o) for o in obs_cfgs), default=3.0))
                for c in candidates
            }
        mean, std, lcb = self._inflated_predictions(candidates, observations, model, problem=problem)
        # Recover log-space sigma conservatively from linear std/mean.
        log_mean = np.log(np.maximum(mean, 1e-9))
        log_sigma = np.sqrt(np.log1p(np.maximum(std, 1e-12) ** 2 / np.maximum(mean, 1e-12) ** 2))
        scores: dict[Any, tuple[float, float, float, float, float, float]] = {}
        local_pool: set[Any] = set()
        if incumbent_cfg is not None:
            local_pool = self._neighbors(incumbent_cfg, [*candidates, *obs_cfgs])
        inc_log = math.log(max(incumbent_ms, 1e-9)) if math.isfinite(incumbent_ms) else float("inf")
        for c, m, s, lm, ls, l in zip(candidates, mean, std, log_mean, log_sigma, lcb):
            sigma = max(float(ls), 0.025)
            if math.isfinite(inc_log):
                z = (inc_log - float(lm)) / sigma
                phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
                Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                ei_log = max(0.0, (inc_log - float(lm)) * Phi + sigma * phi)
                ei = max(0.0, incumbent_ms * ei_log)
            else:
                ei = float(s)
            distance = min((_distance(c, o) for o in obs_cfgs), default=3.0)
            local = 1.0 if c in local_pool else 0.0
            racing = float(ei) + local * max(0.001, 0.55 * (incumbent_ms if math.isfinite(incumbent_ms) else 0.05))
            scores[c] = (float(m), float(s), float(l), float(ei), float(racing), float(distance))
        return scores

    @staticmethod
    def _region_unattempted(region: Region, configs: Sequence[Any], attempted: set[Any]) -> list[Any]:
        return [configs[i] for i in region.indices if configs[i] not in attempted]

    @staticmethod
    def _region_observations(region: Region, configs: Sequence[Any], observations: Sequence[Observation]) -> int:
        cfgs = {configs[i] for i in region.indices}
        return sum(1 for o in observations if o.cfg in cfgs)

    def _region_lcb(
        self,
        region: Region,
        score_by_cfg: dict[Any, tuple[float, float, float, float, float]],
        configs: Sequence[Any],
        attempted: set[Any],
    ) -> float:
        vals = [
            score_by_cfg[configs[i]][2]
            for i in region.indices
            if configs[i] not in attempted and configs[i] in score_by_cfg
        ]
        return min(vals) if vals else float("inf")

    def _prune_regions(
        self,
        tree: SemanticSearchTree,
        configs: Sequence[Any],
        scores: dict[Any, tuple[float, float, float, float, float]],
        observations: Sequence[Observation],
        incumbent: float,
        attempted: set[Any],
        protected_cfg: Any | None,
    ) -> tuple[int, int, list[str]]:
        if not math.isfinite(incumbent) or not scores:
            return 0, 0, []
        protected = {id(r) for r in tree.protected_path(protected_cfg)}
        threshold = incumbent * (1.0 + self.search.prune_margin)
        newly_pruned = 0
        newly_split = 0
        names: list[str] = []

        # Split only after the current region has real hardware witnesses. This keeps
        # the hierarchy coarse where evidence is missing and lets a bad coarse branch
        # disappear after only a few measurements instead of eagerly creating hundreds
        # of tiny leaves with zero evidence.
        leaves = list(tree.active_leaves())
        leaves.sort(key=lambda r: (r.depth, r.key))
        for region in leaves:
            if id(region) in protected or region.pruned:
                continue
            if len(self._region_unattempted(region, configs, attempted)) == 0:
                continue
            robs = self._region_observations(region, configs, observations)
            if robs < self.search.min_region_observations:
                continue
            if len(region.indices) > self.search.min_leaf_size and not region.split:
                children = tree.split(region)
                newly_split += len(children)

        # Recompute the protected path after splitting because newly-created child
        # regions did not exist when the initial protected set was built.
        protected = {id(r) for r in tree.protected_path(protected_cfg)}

        # Evaluate bounds again on the newly created frontier.
        leaves = list(tree.active_leaves())
        for region in leaves:
            if id(region) in protected or region.pruned:
                continue
            remaining = self._region_unattempted(region, configs, attempted)
            if not remaining:
                continue
            robs = self._region_observations(region, configs, observations)
            if robs < self.search.min_region_observations:
                continue
            bound = self._region_lcb(region, scores, configs, attempted)
            if bound >= threshold:
                region.prune_streak += 1
                if region.prune_streak >= max(1, self.search.prune_stability_rounds):
                    tree.mark_pruned(region)
                    newly_pruned += 1
                    names.append(region.key)
            else:
                region.prune_streak = 0

        return newly_pruned, newly_split, names

    def _unpruned_remaining(self, legal: Sequence[Any], attempted: set[Any], tree: SemanticSearchTree) -> list[Any]:
        pruned = tree.pruned_configs()
        return [c for c in legal if c not in attempted and c not in pruned]

    def _witness_candidate(
        self,
        remaining: Sequence[Any],
        observations: Sequence[Observation],
        tree: SemanticSearchTree,
        legal: Sequence[Any],
        attempted: set[Any],
        scores: dict[Any, tuple[float, float, float, float, float, float]],
        protected_cfg: Any | None,
    ) -> Any | None:
        """Pick a point that supplies missing evidence to the shallowest active region."""
        protected = {id(r) for r in tree.protected_path(protected_cfg)}
        regions: list[tuple[int, int, str, Region]] = []
        for region in tree.active_leaves():
            if region.pruned or id(region) in protected:
                continue
            cand = self._region_unattempted(region, legal, attempted)
            if not cand:
                continue
            robs = self._region_observations(region, legal, observations)
            if robs >= self.search.min_region_observations:
                continue
            # Prefer shallow regions first: one witness per coarse branch before
            # spending evaluations on deeper subdivisions.
            regions.append((robs, region.depth, region.key, region))
        if not regions:
            return None
        _, _, _, region = min(regions, key=lambda x: (x[0], x[1], x[2]))
        candidates = [c for c in remaining if c in set(self._region_unattempted(region, legal, attempted))]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda c: (
                scores[c][0],
                -scores[c][3],
                -scores[c][5],
                _cfg_key(c),
            ),
        )

    def _select_batch(
        self,
        legal: Sequence[Any],
        attempted: set[Any],
        observations: Sequence[Observation],
        model: FactorizedGaussianSurrogate | None,
        problem: ProblemKey,
        tree: SemanticSearchTree,
        seed_pool: Sequence[Any],
        incumbent_cfg: Any | None,
        incumbent_ms: float,
        rounds: int,
        last_improvement_round: int,
    ) -> list[Any]:
        remaining = self._unpruned_remaining(legal, attempted, tree)
        if not remaining:
            return []

        pending_seeds = [c for c in seed_pool if c in remaining]
        if pending_seeds:
            # Startup probes can still be batched: they are deliberately diverse.
            return pending_seeds[:min(self.search.batch_size, len(pending_seeds))]

        if len(observations) < self.search.min_model_points or model is None:
            scouts = self._global_scouts(remaining, observations, limit=min(self.search.batch_size, len(remaining)))
            return scouts

        scores = self._candidate_scores(
            remaining, observations, model, problem=problem,
            incumbent_cfg=incumbent_cfg, incumbent_ms=incumbent_ms,
        )
        remaining_set = set(remaining)
        selected: list[Any] = []

        local_pool = set()
        if incumbent_cfg is not None:
            local_pool = self._neighbors(incumbent_cfg, remaining) & remaining_set
        local_ranked = sorted(
            local_pool,
            key=lambda c: (-scores[c][3], -scores[c][4], scores[c][0], scores[c][5], _cfg_key(c)),
        )
        global_ei = sorted(
            remaining,
            key=lambda c: (-scores[c][3], scores[c][0], scores[c][5], _cfg_key(c)),
        )

        plateau = max(0, rounds - last_improvement_round) if last_improvement_round > 0 else 0
        # Early rounds exploit the incumbent basin; after a plateau we deliberately
        # increase global scouting. The proportions are dynamic, not a fixed budget.
        if plateau <= 3:
            quotas = (max(2, self.search.batch_size // 2), 1, 1)
        elif plateau <= 8:
            quotas = (max(1, self.search.batch_size // 3), 2, 1)
        else:
            quotas = (1, max(2, self.search.batch_size // 2), 2)

        def add_unique(seq, limit):
            count = 0
            for c in seq:
                if c not in selected:
                    selected.append(c)
                    count += 1
                    if count >= limit:
                        break

        add_unique(local_ranked, quotas[0])
        add_unique(global_ei, quotas[1])

        # One farthest scout per batch keeps the algorithm from tunnelling forever
        # into a small tile basin. After a plateau, take two.
        scout_limit = 2 if plateau > 8 else 1
        scouts = self._global_scouts([c for c in remaining if c not in selected], observations, limit=min(12, len(remaining)))
        add_unique(scouts, scout_limit)

        if len(selected) < self.search.batch_size:
            # Fill the rest by EI after the explicit lanes, preserving adaptivity.
            add_unique(global_ei, self.search.batch_size - len(selected))
        return selected[:self.search.batch_size]

    @staticmethod
    def _should_refit_model(observation_count: int, last_fit_count: int) -> bool:
        """Keep the expensive surrogate online without refitting on every late point."""
        if observation_count <= 20:
            stride = 1
        elif observation_count <= 48:
            stride = 4
        elif observation_count <= 96:
            stride = 8
        else:
            stride = 12
        return last_fit_count < 0 or observation_count - last_fit_count >= stride

    def run(
        self,
        problem: ProblemKey,
        legal_configs: Sequence[Any],
        evaluate: Callable[[Any, Fidelity], Measurement],
        *,
        static_invalid: Iterable[tuple[Any, str]] = (),
        prior_configs: Sequence[Any] = (),
        use_cache: bool = False,
    ) -> SearchResult:
        legal = list(legal_configs)
        invalid_cfgs = {cfg for cfg, _ in static_invalid}
        space_total = len(legal) + len(invalid_cfgs)
        attempted: set[Any] = set(invalid_cfgs)
        active_legal = [c for c in legal if c not in invalid_cfgs]

        observations: list[Observation] = []
        failures = 0
        correctness_failed = 0
        compile_failures = 0
        cache_hits = 0
        high_fidelity_cache_hits = 0

        # The SQLite cache is the authoritative persistence layer. Every successful
        # result and every deterministic failure for this fingerprint is loaded.
        if use_cache:
            for cfg in active_legal:
                cached = self.cache.get(problem, cfg)
                if cached is None:
                    continue
                attempted.add(cfg)
                cache_hits += 1
                if cached.successful:
                    observations.append(Observation(cfg, cached, -1))
                    if cached.fidelity >= 1:
                        high_fidelity_cache_hits += 1
                else:
                    failures += 1
                    if cached.error_type == "correctness":
                        correctness_failed += 1
                    else:
                        compile_failures += 1

        tree = SemanticSearchTree(active_legal)
        seeds = self._seed_order(active_legal, prior_configs)
        probe, verify = self.search.fidelities()
        best_cfg: Any | None = None
        best_ms = float("inf")
        if observations:
            best_obs = min(observations, key=lambda o: o.measurement.latency_ms or float("inf"))
            best_cfg = best_obs.cfg
            best_ms = float(best_obs.measurement.latency_ms)

        rounds = 0
        new_gpu_evals = 0
        stopped_by_bound = False
        model: FactorizedGaussianSurrogate | None = None
        last_fit_count = -1
        max_rounds = self.search.max_rounds
        last_improvement_round = 0
        last_improvement_obs = 0
        global_scouts_since_improvement = 0
        rounds_without_improvement = 0

        while True:
            if max_rounds is not None and rounds >= max_rounds:
                break

            frontier = self._unpruned_remaining(active_legal, attempted, tree)
            if not frontier:
                if math.isfinite(best_ms):
                    stopped_by_bound = False
                break

            rounds += 1
            if self._should_refit_model(len(observations), last_fit_count):
                model = self._fit_model(observations, problem=problem)
                last_fit_count = len(observations)

            # v12 does not permit surrogate-only global/region pruning. The model is
            # an acquisition function, not a proof that unseen GPU code is bad.
            frontier = self._unpruned_remaining(active_legal, attempted, tree)

            seed_batch = [c for c in seeds if c not in attempted and c in self._unpruned_remaining(active_legal, attempted, tree)]
            # Adaptive convergence: no hard evaluation budget. The search may finish
            # only after a sustained plateau, a well-explored incumbent neighbourhood,
            # several explicit global scouts, and negligible modelled improvement.
            # Unlike v10, this is not a global LCB proof over every unseen point.
            if (model is not None and len(observations) >= max(24, self.search.min_model_points * 2)
                    and best_cfg is not None and last_improvement_obs > 0):
                plateau_evals = len(observations) - last_improvement_obs
                local_all = set(self._neighbors(best_cfg, active_legal))
                local_unseen = [c for c in local_all if c in frontier and c not in attempted]
                local_coverage = 1.0 if not local_all else 1.0 - (len(local_unseen) / len(local_all))
                scores_now = self._candidate_scores(
                    frontier, observations, model, problem=problem,
                    incumbent_cfg=best_cfg, incumbent_ms=best_ms,
                )
                top_ei = sorted((scores_now[c][3] for c in frontier), reverse=True)[:8]
                best_ei = top_ei[0] if top_ei else 0.0
                min_pred_mean = min((scores_now[c][0] for c in frontier), default=float("inf"))
                stop_ei = max(0.00045, 0.02 * best_ms)
                if plateau_evals >= 8:
                    # Stop when the model has converged around the incumbent: the
                    # predicted best unseen point is not materially better, its EI is
                    # small, and the local basin plus several global scouts were tested.
                    if (min_pred_mean >= best_ms * 1.01
                            and best_ei <= stop_ei
                            and local_coverage >= 0.40
                            and global_scouts_since_improvement >= 2):
                        print(
                            f"adaptive stop: converged plateau_evals={plateau_evals} "
                            f"local_coverage={local_coverage:.0%} global_scouts={global_scouts_since_improvement} "
                            f"min_pred={min_pred_mean:.6f} ms best_EI={best_ei:.6f} ms"
                        )
                        break

            batch = self._select_batch(
                active_legal, attempted, observations, model, problem, tree, seed_batch, best_cfg, best_ms,
                rounds, last_improvement_round,
            )
            if not batch:
                # If an implementation corner leaves no selectable leaf, split one
                # real semantic region rather than silently abandoning the space.
                leaves = [r for r in tree.active_leaves() if self._region_unattempted(r, active_legal, attempted)]
                leaves.sort(key=lambda r: (len(r.indices), r.key))
                if leaves:
                    children = tree.split(leaves[-1])
                    if children:
                        continue
                break

            batch_start_best = best_ms
            scout_candidates = set()
            if model is not None and best_cfg is not None and last_improvement_round != rounds:
                scout_candidates = set(self._global_scouts(
                    [c for c in active_legal if c not in attempted], observations, limit=12
                ))
            for cfg in batch:
                if cfg in attempted:
                    continue
                attempted.add(cfg)
                try:
                    measurement = evaluate(cfg, probe)
                except Exception as exc:
                    measurement = Measurement.failed(type(exc).__name__, str(exc), fidelity=probe.level)
                if use_cache:
                    self.cache.put(problem, cfg, measurement)
                new_gpu_evals += 1
                if measurement.successful:
                    observations.append(Observation(cfg, measurement, rounds))
                    if cfg in scout_candidates and last_improvement_round != rounds:
                        global_scouts_since_improvement += 1
                    if measurement.latency_ms is not None and measurement.latency_ms < best_ms:
                        best_ms = float(measurement.latency_ms)
                        best_cfg = cfg
                        last_improvement_round = rounds
                        last_improvement_obs = len(observations)
                        global_scouts_since_improvement = 0
                        print(f"NEW INCUMBENT {best_ms:.4f} ms {cfg}")
                        # A real improvement is new information. Stop the stale batch
                        # immediately so the next candidate is chosen from the updated
                        # incumbent rather than spending compile time on old scores.
                        if math.isfinite(batch_start_best) and best_ms < batch_start_best * 0.985:
                            break
                else:
                    failures += 1
                    if measurement.error_type == "correctness":
                        correctness_failed += 1
                    else:
                        compile_failures += 1
                    print(f"    FAILED {cfg}: {measurement.error_type}: {measurement.error_message}")

            seeds = [c for c in seeds if c not in attempted]
            if last_improvement_round != rounds:
                rounds_without_improvement += 1
            else:
                rounds_without_improvement = 0
            print(
                f"adaptive: round={rounds} new_evals={new_gpu_evals} "
                f"cached={cache_hits} successful={len(observations)} "
                f"best={'%.4f' % best_ms if math.isfinite(best_ms) else 'n/a'} "
                f"frontier={len(self._unpruned_remaining(active_legal, attempted, tree))}"
            )

        # High-fidelity verification is not part of the exploration budget. It runs
        # only for candidates without an already high-fidelity cached measurement.
        model = self._fit_model(observations, problem=problem)
        frontier = self._unpruned_remaining(active_legal, attempted, tree)
        candidate_set: list[Any] = []
        if model is not None and frontier and self.search.verification_topk > 0:
            scores = self._candidate_scores(frontier, observations, model, problem=problem, incumbent_cfg=best_cfg, incumbent_ms=best_ms)
            candidate_set.extend(sorted(frontier, key=lambda c: (scores[c][3], scores[c][0], _cfg_key(c)))[:self.search.verification_topk])
        for o in sorted(observations, key=lambda x: (x.measurement.latency_ms or float("inf"), _cfg_key(x.cfg)))[:self.search.verification_topk]:
            if o.cfg not in candidate_set:
                candidate_set.append(o.cfg)
        verified = 0
        verification_gpu_evals = 0
        for cfg in candidate_set:
            cached = self.cache.get(problem, cfg) if use_cache else None
            if cached is not None and cached.successful and cached.fidelity >= verify.level:
                continue
            try:
                measurement = evaluate(cfg, verify)
            except Exception as exc:
                measurement = Measurement.failed(type(exc).__name__, str(exc), fidelity=verify.level)
            if use_cache:
                self.cache.put(problem, cfg, measurement)
            verification_gpu_evals += 1
            if measurement.successful:
                verified += 1
                observations[:] = [o for o in observations if o.cfg != cfg]
                observations.append(Observation(cfg, measurement, rounds + 1))
                if measurement.latency_ms is not None and measurement.latency_ms < best_ms:
                    best_ms = float(measurement.latency_ms)
                    best_cfg = cfg
                    print(f"NEW INCUMBENT verified {best_ms:.4f} ms {cfg}")
            else:
                failures += 1
                if measurement.error_type == "correctness":
                    correctness_failed += 1
                else:
                    compile_failures += 1

        pruned_cfgs = tree.pruned_configs()
        pruned_cfgs.difference_update(attempted)
        active_untested = [c for c in active_legal if c not in attempted and c not in pruned_cfgs]
        return SearchResult(
            best_config=best_cfg,
            best_latency_ms=None if not math.isfinite(best_ms) else best_ms,
            space_total=space_total,
            legal_total=len(active_legal),
            static_invalid=len(invalid_cfgs),
            attempted=len([c for c in attempted if c in active_legal]),
            successful=len({o.cfg for o in observations if o.measurement.successful}),
            compile_failures=compile_failures,
            correctness_failures=correctness_failed,
            model_pruned=len(pruned_cfgs),
            region_pruned=sum(1 for n in tree.nodes if n.pruned),
            untested=len([c for c in active_legal if c not in attempted]),
            adaptive_search=True,
            hierarchical_pruning=False,
            stratified_acquisition=True,
            search_engine=self.ENGINE,
            stopped_by_global_bound=stopped_by_bound,
            rounds=rounds,
            metadata={
                "cache_hits": cache_hits,
                "high_fidelity_cache_hits": high_fidelity_cache_hits,
                "new_gpu_evals": new_gpu_evals,
                "verification_gpu_evals": verification_gpu_evals,
                "verified_candidates": verified,
                "tree_nodes": len(tree.nodes),
                "active_untested": len(active_untested),
                "probe_reps": probe.rep,
                "verification_reps": verify.rep,
                "unlimited_frontier": self.search.max_rounds is None,
                "search_note": "Fresh online measurement is primary; no surrogate-only region pruning or global-LCB termination is used. Adaptive convergence may stop after a sustained plateau; no max-evals budget is used.",
            },
        )
