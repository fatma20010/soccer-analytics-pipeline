from soccer_analytics.modules.possession import PossessionTracker

def test_initial_state():
    p = PossessionTracker()
    assert p.get_possession() == {}
