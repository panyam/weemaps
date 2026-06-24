# Weewar Game JSON Parser - Development History

## Overview

This document describes the analysis and parsing of Weewar game JSON data files, which contain complete game state and turn history for played games.

## JSON Format Analysis

### General Structure

The Weewar JSON uses a **key-data compression pattern** throughout to reduce file size. Instead of repeated object keys, the format stores:
- A `key` array defining field names
- A `data` array with values in matching index order

Example:
```json
{
  "key": ["x", "y", "health"],
  "data": [[1, 2, 10], [3, 4, 8]]
}
```
Represents two objects: `{x: 1, y: 2, health: 10}` and `{x: 3, y: 4, health: 8}`

### Top-Level Sections

| Section | Purpose |
|---------|---------|
| `time` | Timestamp of the game data |
| `game` | Game metadata and complete turn history |
| `player` | Player information (usernames, stats, status) |
| `coordinate` | Current board state (all units and positions) |
| `chat` | Game chat and event log |

---

## The `game` Section

### Metadata Fields

```json
"key": ["started", "finished", "ranked", "timeLimit", "round", "turn"]
```

| Field | Type | Description |
|-------|------|-------------|
| `started` | bool | Whether the game has begun |
| `finished` | bool | Whether the game is complete |
| `ranked` | bool | Whether it's a ranked match |
| `timeLimit` | int | Time limit in minutes (1440 = 24 hours) |
| `round` | int | Current round number |
| `turn` | array | Nested array of all turn data |

### Turn Structure

The `turn` field uses a nested array hierarchy:

```
turn[round_index][player_index][action_index] = action_tuple
```

- **Round index**: 0-based index for each round
- **Player index**: 0 = first player, 1 = second player
- **Action index**: Sequential actions within that player's turn

---

## Action Types

Four action types were identified, each with a distinct tuple structure:

### Build (`"b"`)
Creates a new unit at a base/factory location.

```
["b", x, y, unit_type, tone]
```

| Index | Field | Description |
|-------|-------|-------------|
| 0 | `"b"` | Action type identifier |
| 1 | x | X coordinate of build location |
| 2 | y | Y coordinate of build location |
| 3 | unit_type | Integer ID of unit type |
| 4 | tone | Visual variant/skin of the unit |

### Move (`"m"`)
Moves a unit from one tile to another.

```
["m", from_x, from_y, to_x, to_y]
```

| Index | Field | Description |
|-------|-------|-------------|
| 0 | `"m"` | Action type identifier |
| 1 | from_x | Starting X coordinate |
| 2 | from_y | Starting Y coordinate |
| 3 | to_x | Destination X coordinate |
| 4 | to_y | Destination Y coordinate |

### Attack (`"a"`)
Combat action between units.

```
["a", attacker_x, attacker_y, target_x, target_y, damage_dealt, damage_received, combat_details, extra_info]
```

| Index | Field | Description |
|-------|-------|-------------|
| 0 | `"a"` | Action type identifier |
| 1 | attacker_x | Attacker's X coordinate |
| 2 | attacker_y | Attacker's Y coordinate |
| 3 | target_x | Target's X coordinate |
| 4 | target_y | Target's Y coordinate |
| 5 | damage_dealt | Damage inflicted on defender |
| 6 | damage_received | Counter-attack damage to attacker |
| 7 | combat_details | Array of combat modifiers (terrain, etc.) |
| 8 | extra_info | Additional data (often null) |

The `combat_details` array appears to contain combat modifiers like terrain defense bonuses, but the exact meaning of each index is not yet fully decoded.

### Heal/Hold (`"h"`)
Repair action or skip/hold in place.

```
["h", x, y]
```

| Index | Field | Description |
|-------|-------|-------------|
| 0 | `"h"` | Action type identifier |
| 1 | x | X coordinate of unit |
| 2 | y | Y coordinate of unit |

---

## The `coordinate` Section (Board State)

Contains all units currently on the board.

```json
"key": ["x", "y", "unitId", "unitPlayer", "tilePlayer", "unitHealth", 
        "unitAttackCount", "unitProgression", "unitTone"]
```

| Field | Description |
|-------|-------------|
| `x`, `y` | Unit position on the board |
| `unitId` | Unit type identifier |
| `unitPlayer` | Owner player (1-indexed: 1 or 2) |
| `tilePlayer` | Owner of the tile (for bases/cities), null otherwise |
| `unitHealth` | Current HP (max is 10) |
| `unitAttackCount` | Number of attacks made this turn |
| `unitProgression` | Available actions remaining (nested arrays) |
| `unitTone` | Visual variant/skin |

### Data Format Variants

Unit data can appear in two formats interchangeably:

**Array format:**
```json
[2, 4, 8, 2, null, 10, 0, [...], 3]
```

**Object format (sparse representation):**
```json
{"0": 3, "1": 3, "2": 2, "3": 2, "4": null, "5": 10, "6": 0, "7": [...], "8": 2}
```

The parser normalizes both formats to a consistent dictionary.

---

## The `player` Section

```json
"key": ["active", "defeated", "accepted", "draw", "online", "avatar", 
        "badgeId", "badgeTitle", "proImage", "proTitle", "rank", "rankChange",
        "coin", "team", "skipCount", "turnDate", "inviteDate", "playerId", 
        "username", "unitSkin"]
```

Key fields:
- `username`: Display name
- `playerId`: Unique player ID (-1 for AI)
- `active`: Currently active player's turn
- `defeated`: Player has lost
- `coin`: Current coin/resource count

---

## Important Indexing Discovery

**Critical finding:** The `unitPlayer` field in coordinate data uses **1-indexed** values (1, 2), while the player data array is **0-indexed** (0, 1).

When looking up a player from a unit's `owner_player` field, subtract 1:
```python
player_index = unit.owner_player - 1
```

The parser includes a helper method `get_player_by_unit_owner()` to handle this conversion.

---

## Parser Implementation

### Data Classes Created

**Actions:**
- `BuildAction` - Unit creation
- `MoveAction` - Unit movement
- `AttackAction` - Combat with damage tracking
- `HealAction` - Repair/hold

**Game Structure:**
- `WeewarGame` - Top-level container
- `GameState` - Metadata + rounds list
- `Round` - Contains player turns
- `PlayerTurn` - List of actions for one player

**Board State:**
- `Player` - Player metadata
- `Unit` - Unit position and state
- `ChatMessage` - Chat/event log entries

### Usage

```python
from weewar_parser import parse_weewar_json

# Parse from JSON string or dict
game = parse_weewar_json(json_data)

# Access turn history
for round in game.game_state.rounds:
    for player_turn in round.player_turns:
        for action in player_turn.actions:
            print(action)

# Get player info
player = game.get_player_by_unit_owner(unit.owner_player)

# Pretty print game log
game.print_game_log()
```

---

## Future Work

Potential enhancements:
- Unit type ID to name mapping (e.g., 1 = Infantry, 2 = Tank, etc.)
- Combat details array decoding
- Game replay/simulation from turn data
- Export to other formats (PGN-style notation, etc.)
- Board visualization