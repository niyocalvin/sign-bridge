"""Stability rule for live letters: a letter is typed only after it has been the confident prediction for `n` frames in a row.

After a letter is typed the same letter is locked until the hand changes (another confident letter, or the hand is lost for
`release` frames), so holding a sign does not type it repeatedly.
"""


class Debouncer:
    def __init__(self, n=12, min_conf=0.6, release=3, adaptive=False):
        """adaptive (fast mode): the number of frames needed shrinks when every frame of the streak was very confident:
        all >= 0.95 -> n/4 (at least 3), all >= 0.85 -> n/2 (at least 4), otherwise n."""
        self.n, self.min_conf, self.release, self.adaptive = n, min_conf, release, adaptive
        self.cand, self.count, self.locked, self.lost, self.streak_min = None, 0, None, 0, 1.0

    def reset(self):
        self.cand, self.count, self.locked, self.lost, self.streak_min = None, 0, None, 0, 1.0

    def needed(self):
        if not self.adaptive:
            return self.n
        if self.streak_min >= 0.95:
            return min(self.n, max(3, round(self.n / 4)))
        if self.streak_min >= 0.85:
            return min(self.n, max(4, round(self.n / 2)))
        return self.n

    def update(self, letter, conf):
        """letter=None means no hand. Returns (progress 0..1, committed letter or None)."""
        if letter is None:
            self.lost += 1
            self.cand, self.count = None, 0
            if self.lost >= self.release:
                self.locked = None
            return 0.0, None
        self.lost = 0
        if conf < self.min_conf:                 # unsure frame: break the streak
            self.cand, self.count = None, 0
            return 0.0, None
        if letter != self.locked and self.locked is not None:
            self.locked = None                   # hand changed to another letter: unlock
        if letter == self.locked:
            return 1.0, None
        if letter == self.cand:
            self.count += 1
            self.streak_min = min(self.streak_min, conf)
        else:
            self.cand, self.count, self.streak_min = letter, 1, conf
        need = self.needed()
        if self.count >= need:
            self.locked, self.cand, self.count, self.streak_min = letter, None, 0, 1.0
            return 1.0, letter
        return self.count / need, None
