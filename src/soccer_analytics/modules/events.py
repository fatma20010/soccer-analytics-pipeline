from __future__ import annotations
from typing import List, Dict, Optional
import time

class MatchEvents:
    """Tracks cards, fouls/free kicks, penalties. Initially manual triggers via keyboard.
    Structure can be extended to automatic classifiers.
    """
    def __init__(self):
        self.cards: List[dict] = []  # {time, track_id, team, type}
        self.free_kicks: List[dict] = []  # {time, team, location?(optional)}
        self.penalties: List[dict] = []  # {time, team}

    def add_card(self, track_id: int, team: int | None, card_type: str):
        self.cards.append({'timestamp': time.time(), 'track_id': track_id, 'team': team, 'type': card_type})

    def add_free_kick(self, team: int | None):
        self.free_kicks.append({'timestamp': time.time(), 'team': team})

    def add_penalty(self, team: int | None):
        self.penalties.append({'timestamp': time.time(), 'team': team})

    def summary(self):
        reds = sum(1 for c in self.cards if c['type']=='R')
        yellows = sum(1 for c in self.cards if c['type']=='Y')
        return {
            'yellow_cards': yellows,
            'red_cards': reds,
            'free_kicks': len(self.free_kicks),
            'penalties': len(self.penalties)
        }
