class NetworkProfile:
    """
    Link model in bytes/sec.
    - base: typical (median-ish) link rate
    - jitter: log-normal multiplicative noise, changes every two rounds
    - one client (cid=0) is throttled after round 3 to a fixed fraction of base
    """
    def __init__(self, seed: int = 42, base: float = 4e5, throttle_frac: float = 0.25, floor_bps: float = 1e4):
        self.seed = int(seed)
        self.base = float(base)                 # e.g., 4e5 B/s ≈ 0.38 MB/s
        self.throttle_frac = float(throttle_frac)  # 0.25× base → strong straggler
        self.floor_bps = float(floor_bps)       # never below 10 KB/s

    def _seg_key(self, cid: int, rnd: int) -> int:
        segment = max(0, (int(rnd) - 1) // 2)   # changes every two rounds
        return (self.seed * 10007) ^ (cid * 997) ^ segment

    def _lognormal_factor(self, key: int, mu: float = 0.0, sigma: float = 0.30) -> float:
        import math, random
        rng = random.Random(key)
        return math.exp(rng.gauss(mu, sigma))   # multiplicative jitter

    def uplink(self, cid: int, rnd: int) -> float:
        key = self._seg_key(cid, rnd)
        rate = self.base * self._lognormal_factor(key)
        # Throttle client 0 after round 3 to (throttle_frac × base)
        if cid == 0 and rnd >= 3:
            cap = max(self.floor_bps, self.throttle_frac * self.base)
            rate = min(rate, cap)
        return max(self.floor_bps, rate)

    def downlink(self, cid: int, rnd: int) -> float:
        key = self._seg_key(cid, rnd) ^ 0x9E3779B9  # decorrelate from uplink
        rate = self.base * self._lognormal_factor(key)
        # (Optionally also throttle downlink for cid=0; usually not necessary)
        return max(self.floor_bps, rate)