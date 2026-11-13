"""
Unit tests for TacticalRoster module
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.analytics.roster import TacticalRoster


def test_tactical_roster_basic():
    # Create synthetic kinematics dataframe with 26 players (simulating real match scenario)
    records = []
    for i in range(1, 27):
        records.append({
            "Player ID": f"#{i}",
            "Frames Tracked": 30 if i <= 22 else 5,
            "Total Distance (px)": 50.0 + i * 5,
            "Average Speed (px/s)": 30.0 + i,
            "Max Speed (px/s)": 60.0 + i * 2,
            "Tracking Duration (s)": 1.2
        })
    df_kin = pd.DataFrame(records)

    # Tracking coordinates with synthetic x positions
    track_records = []
    for i in range(1, 27):
        track_records.append({
            "track_id": i,
            "x": 100.0 + (i * 40.0),
            "y": 300.0
        })
    df_track = pd.DataFrame(track_records)

    # Team assignments: 13 White, 13 Red
    player_teams = {i: ("Team White" if i % 2 == 1 else "Team Red") for i in range(1, 27)}

    match_roster, officials_df = TacticalRoster.build_tactical_roster(df_kin, df_track, player_teams)

    # Must contain 22 official field players (11 White + 11 Red)
    assert len(match_roster) == 22
    assert len(match_roster[match_roster["Team"] == "Team White"]) == 11
    assert len(match_roster[match_roster["Team"] == "Team Red"]) == 11

    # Check tactical positions
    white_1 = match_roster[(match_roster["Team"] == "Team White") & (match_roster["Jersey Number"] == 1)].iloc[0]
    assert white_1["Tactical Role"] == "Goalkeeper"
    assert "White #1" in white_1["Tactical Jersey"]

    # Check peripheral tracks
    assert len(officials_df) == 4

    # Test display mapping
    mapping = TacticalRoster.get_player_display_mapping(match_roster, officials_df)
    assert len(mapping) == 26
    assert 1 in mapping
    assert "⚪ White #1" in mapping[1]


def test_tactical_roster_empty():
    empty_kin = pd.DataFrame()
    roster, off = TacticalRoster.build_tactical_roster(empty_kin)
    assert roster.empty
    assert off.empty
