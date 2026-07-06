#!/bin/bash
set -e  # Exit on first error

# Error handler - show which command failed
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR

# Lilbattle Game Replay Script
# Generated from game history
# Players: golilbattle, Computer (A)

# Give user the chance to set the world id here
WORLD_ID="32112070"
# Look for the line that has "export LILBATTLE_GAME_ID=...." here so we can extract gameid from it
gameIdLine=$(ww new --game-income 100 --landbase-income 100 $WORLD_ID | grep "export LILBATTLE_GAME_ID")
gameId=$(echo $gameIdLine | sed -e "s/.*export.LILBATTLE_GAME_ID=//g")
export LILBATTLE_GAME_ID=$gameId
export LILBATTLE_CONFIRM=false
echo Created game for testing: $LILBATTLE_GAME_ID
if [ -n "$LILBATTLE_SERVER" ]; then echo "Game viewable at: ${LILBATTLE_SERVER%/api}/games/$LILBATTLE_GAME_ID/view"; fi

# ============================================================
# Round 1
# ============================================================

# --- Player 0 (golilbattle) ---
ww move 1,1 0,3
ww assert exists unit 0,3
ww assert notexists unit 1,1
ww endturn

# --- Player 1 (Computer (A)) ---
ww build t:3,6 1
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 1]
ww move 5,5 4,3
ww assert exists unit 4,3
ww assert notexists unit 5,5
ww move 4,6 5,3
ww assert exists unit 5,3
ww assert notexists unit 4,6
ww endturn

# ============================================================
# Round 2
# ============================================================

# --- Player 0 (golilbattle) ---
ww build t:2,1 2
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 2]
ww capture 0,3
ww endturn

# --- Player 1 (Computer (A)) ---
ww move 3,6 2,4
ww assert exists unit 2,4
ww assert notexists unit 3,6
ww build t:3,6 1
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 1]
ww move 5,3 4,2
ww assert exists unit 4,2
ww assert notexists unit 5,3
ww move 4,3 5,2
ww assert exists unit 5,2
ww assert notexists unit 4,3
ww endturn

# ============================================================
# Round 3
# ============================================================

# --- Player 0 (golilbattle) ---
ww move 2,1 3,1
ww assert exists unit 3,1
ww assert notexists unit 2,1
ww build t:2,1 2
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 2]
ww endturn

# --- Player 1 (Computer (A)) ---
ww move 4,2 4,3
ww assert exists unit 4,3
ww assert notexists unit 4,2
ww move 2,4 4,2
ww assert exists unit 4,2
ww assert notexists unit 2,4
ww move 3,6 5,3
ww assert exists unit 5,3
ww assert notexists unit 3,6
ww build t:3,6 2
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 2]
ww move 5,2 3,3
ww assert exists unit 3,3
ww assert notexists unit 5,2
ww endturn

# ============================================================
# Round 4
# ============================================================

# --- Player 0 (golilbattle) ---
ww build t:0,3 8
ww assert exists unit 0,3
ww assert unit 0,3 [type eq 8]
ww move 2,1 1,2
ww assert exists unit 1,2
ww assert notexists unit 2,1
ww build t:2,1 2
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 2]
ww move 3,1 3,0
ww assert exists unit 3,0
ww assert notexists unit 3,1
ww endturn

# --- Player 1 (Computer (A)) ---
ww move 4,3 4,1
ww assert exists unit 4,1
ww assert notexists unit 4,3
ww move 5,3 5,2
ww assert exists unit 5,2
ww assert notexists unit 5,3
ww move 3,3 5,1
ww assert exists unit 5,1
ww assert notexists unit 3,3
ww move 3,6 4,4
ww assert exists unit 4,4
ww assert notexists unit 3,6
ww build t:3,6 1
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 1]
ww move 4,2 3,3
ww assert exists unit 3,3
ww assert notexists unit 4,2
ww endturn

# ============================================================
# Round 5
# ============================================================

# --- Player 0 (golilbattle) ---
ww attack 0,3 3,3
ww assert unit 3,3 [health lte 4]
ww move 2,1 2,0
ww assert exists unit 2,0
ww assert notexists unit 2,1
ww build t:2,1 2
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 2]
ww move 1,2 0,4
ww assert exists unit 0,4
ww assert notexists unit 1,2
ww move 3,0 3,1
ww assert exists unit 3,1
ww assert notexists unit 3,0
ww attack 3,1 4,1
ww assert unit 4,1 [health lte 4]
ww assert unit 3,1 [health lte 5]
ww endturn

# --- Player 1 (Computer (A)) ---
ww move 5,2 4,2
ww assert exists unit 4,2
ww assert notexists unit 5,2
ww move 4,4 4,3
ww assert exists unit 4,3
ww assert notexists unit 4,4
ww move 4,1 3,2
ww assert exists unit 3,2
ww assert notexists unit 4,1
# Heal/Hold at 3,3
ww attack 3,2 3,1
ww assert unit 3,1 [health lte 9]
ww assert unit 3,2 [health lte 7]
ww move 5,1 5,2
ww assert exists unit 5,2
ww assert notexists unit 5,1
ww move 3,6 2,4
ww assert exists unit 2,4
ww assert notexists unit 3,6
ww build t:3,6 2
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 2]
ww endturn

# ============================================================
# Round 6
# ============================================================

# --- Player 0 (golilbattle) ---
ww move 0,3 1,2
ww assert exists unit 1,2
ww assert notexists unit 0,3
ww build t:0,3 2
ww assert exists unit 0,3
ww assert unit 0,3 [type eq 2]
ww move 2,1 3,0
ww assert exists unit 3,0
ww assert notexists unit 2,1
ww build t:2,1 2
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 2]
ww move 2,0 2,2
ww assert exists unit 2,2
ww assert notexists unit 2,0
ww attack 2,2 3,2
ww assert unit 3,2 [health lte 3]
# Heal/Hold at 3,1
ww move 0,4 1,3
ww assert exists unit 1,3
ww assert notexists unit 0,4
ww endturn

# --- Player 1 (Computer (A)) ---
ww move 5,2 4,1
ww assert exists unit 4,1
ww assert notexists unit 5,2
ww move 3,6 4,4
ww assert exists unit 4,4
ww assert notexists unit 3,6
ww build t:3,6 2
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 2]
ww move 3,3 5,2
ww assert exists unit 5,2
ww assert notexists unit 3,3
ww move 4,3 5,1
ww assert exists unit 5,1
ww assert notexists unit 4,3
ww move 2,4 4,3
ww assert exists unit 4,3
ww assert notexists unit 2,4
ww attack 4,1 3,1
ww assert unit 3,1 [health lte 7]
ww assert unit 4,1 [health lte 8]
ww move 4,2 3,3
ww assert exists unit 3,3
ww assert notexists unit 4,2
ww endturn

# ============================================================
# Round 7
# ============================================================

# --- Player 0 (golilbattle) ---
ww attack 1,2 4,1
ww assert unit 4,1 [health lte 5]
ww move 2,2 2,3
ww assert exists unit 2,3
ww assert notexists unit 2,2
ww attack 2,3 3,3
ww assert unit 3,3 [health lte 3]
ww assert unit 2,3 [health lte 8]
ww move 1,3 2,2
ww assert exists unit 2,2
ww assert notexists unit 1,3
ww move 0,3 1,3
ww assert exists unit 1,3
ww assert notexists unit 0,3
ww move 2,1 3,2
ww assert exists unit 3,2
ww assert notexists unit 2,1
ww attack 3,2 3,3
ww assert unit 3,3 [health lte 4]
ww assert unit 3,2 [health lte 9]
ww attack 3,1 4,1
ww assert unit 3,1 [health lte 9]
ww move 3,0 3,1
ww assert exists unit 3,1
ww assert notexists unit 3,0
ww attack 3,1 4,1
ww assert unit 4,1 [health lte 3]
ww assert unit 3,1 [health lte 9]
ww build t:2,1 8
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 8]
ww build t:0,3 2
ww assert exists unit 0,3
ww assert unit 0,3 [type eq 2]
ww endturn

# --- Player 1 (Computer (A)) ---
ww move 4,4 4,2
ww assert exists unit 4,2
ww assert notexists unit 4,4
ww move 4,3 3,3
ww assert exists unit 3,3
ww assert notexists unit 4,3
# Heal/Hold at 5,2
ww move 5,1 4,1
ww assert exists unit 4,1
ww assert notexists unit 5,1
ww move 3,6 4,4
ww assert exists unit 4,4
ww assert notexists unit 3,6
ww build t:3,6 2
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 2]
ww attack 4,2 3,2
ww assert unit 3,2 [health lte 5]
ww assert unit 4,2 [health lte 7]
ww attack 4,1 3,2
ww assert unit 3,2 [health lte 5]
ww assert unit 4,1 [health lte 9]
ww attack 3,3 2,3
ww assert unit 2,3 [health lte 8]
ww assert unit 3,3 [health lte 5]
ww endturn

# ============================================================
# Round 8
# ============================================================

# --- Player 0 (golilbattle) ---
ww attack 1,2 4,1
ww assert unit 4,1 [health lte 6]
ww attack 2,1 4,1
ww assert unit 4,1 [health lte 6]
ww move 2,2 3,2
ww assert exists unit 3,2
ww assert notexists unit 2,2
ww attack 3,2 4,2
ww assert unit 4,2 [health lte 5]
ww assert unit 3,2 [health lte 7]
ww attack 2,3 3,3
ww assert unit 3,3 [health lte 6]
ww assert unit 2,3 [health lte 9]
ww move 3,1 4,2
ww assert exists unit 4,2
ww assert notexists unit 3,1
ww attack 4,2 3,3
ww assert unit 3,3 [health lte 5]
ww move 0,3 1,4
ww assert exists unit 1,4
ww assert notexists unit 0,3
ww move 1,3 2,4
ww assert exists unit 2,4
ww assert notexists unit 1,3
ww build t:0,3 5
ww assert exists unit 0,3
ww assert unit 0,3 [type eq 5]
ww endturn

# --- Player 1 (Computer (A)) ---
ww move 5,2 4,3
ww assert exists unit 4,3
ww assert notexists unit 5,2
ww move 4,4 5,3
ww assert exists unit 5,3
ww assert notexists unit 4,4
ww move 3,6 4,4
ww assert exists unit 4,4
ww assert notexists unit 3,6
ww build t:3,6 5
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 5]
ww attack 4,3 4,2
ww assert unit 4,2 [health lte 8]
ww assert unit 4,3 [health lte 6]
ww endturn

# ============================================================
# Round 9
# ============================================================

# --- Player 0 (golilbattle) ---
ww move 0,3 3,3
ww assert exists unit 3,3
ww assert notexists unit 0,3
ww attack 3,3 4,3
ww assert unit 4,3 [health lte 3]
ww move 2,1 5,1
ww assert exists unit 5,1
ww assert notexists unit 2,1
ww move 1,2 1,5
ww assert exists unit 1,5
ww assert notexists unit 1,2
ww build t:0,3 2
ww assert exists unit 0,3
ww assert unit 0,3 [type eq 2]
ww build t:2,1 8
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 8]
ww move 2,4 3,4
ww assert exists unit 3,4
ww assert notexists unit 2,4
ww attack 3,4 4,4
ww assert unit 4,4 [health lte 5]
ww assert unit 3,4 [health lte 7]
ww move 1,4 2,4
ww assert exists unit 2,4
ww assert notexists unit 1,4
# Heal/Hold at 4,2
# Heal/Hold at 3,2
# Heal/Hold at 2,3
ww endturn

# --- Player 1 (Computer (A)) ---
ww move 3,6 4,3
ww assert exists unit 4,3
ww assert notexists unit 3,6
ww build t:3,6 8
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 8]
ww attack 4,3 4,2
ww assert unit 4,2 [health lte 5]
ww assert unit 4,3 [health lte 8]
ww move 4,4 3,5
ww assert exists unit 3,5
ww assert notexists unit 4,4
ww move 5,3 5,2
ww assert exists unit 5,2
ww assert notexists unit 5,3
ww attack 3,5 3,4
ww assert unit 3,4 [health lte 9]
ww assert unit 3,5 [health lte 6]
ww attack 5,2 4,2
ww assert unit 4,2 [health lte 4]
ww endturn

# ============================================================
# Round 10
# ============================================================

# --- Player 0 (golilbattle) ---
ww attack 5,1 4,3
ww assert unit 4,3 [health lte 6]
ww attack 3,3 4,3
ww assert unit 4,3 [health lte 7]
ww attack 3,4 4,3
ww assert unit 4,3 [health lte 6]
ww move 2,4 2,5
ww assert exists unit 2,5
ww assert notexists unit 2,4
ww attack 2,5 3,5
ww assert unit 3,5 [health lte 5]
ww move 0,3 1,4
ww assert exists unit 1,4
ww assert notexists unit 0,3
ww attack 1,5 3,6
ww assert unit 3,6 [health lte 5]
ww assert unit 1,5 [health lte 5]
ww move 2,1 3,1
ww assert exists unit 3,1
ww assert notexists unit 2,1
# Heal/Hold at 2,3
# Heal/Hold at 3,2
ww build t:2,1 2
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 2]
ww build t:0,3 8
ww assert exists unit 0,3
ww assert unit 0,3 [type eq 8]
ww endturn

# --- Player 1 (Computer (A)) ---
ww move 3,6 5,3
ww assert exists unit 5,3
ww assert notexists unit 3,6
ww build t:3,6 5
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 5]
ww move 5,2 4,2
ww assert exists unit 4,2
ww assert notexists unit 5,2
ww attack 4,2 5,1
ww assert unit 5,1 [health lte 4]
ww endturn

# ============================================================
# Round 11
# ============================================================

# --- Player 0 (golilbattle) ---
ww attack 3,3 4,2
ww assert unit 4,2 [health lte 4]
ww assert unit 3,3 [health lte 6]
ww attack 3,2 4,2
ww assert unit 4,2 [health lte 6]
ww assert unit 3,2 [health lte 9]
ww attack 5,1 5,3
ww assert unit 5,3 [health lte 9]
ww assert unit 5,1 [health lte 8]
ww attack 1,5 3,6
ww assert unit 3,6 [health lte 9]
ww move 2,5 3,5
ww assert exists unit 3,5
ww assert notexists unit 2,5
ww attack 3,5 3,6
ww assert unit 3,6 [health lte 4]
ww assert unit 3,5 [health lte 3]
ww move 3,4 2,6
ww assert exists unit 2,6
ww assert notexists unit 3,4
ww attack 2,6 3,6
ww assert unit 3,6 [health lte 6]
ww assert unit 2,6 [health lte 8]
ww move 2,3 4,3
ww assert exists unit 4,3
ww assert notexists unit 2,3
ww attack 4,3 5,3
ww assert unit 5,3 [health lte 5]
ww move 1,4 2,4
ww assert exists unit 2,4
ww assert notexists unit 1,4
ww move 3,1 4,2
ww assert exists unit 4,2
ww assert notexists unit 3,1
ww move 2,1 4,1
ww assert exists unit 4,1
ww assert notexists unit 2,1
ww move 0,3 1,4
ww assert exists unit 1,4
ww assert notexists unit 0,3
ww build t:0,3 3
ww assert exists unit 0,3
ww assert unit 0,3 [type eq 3]
ww build t:2,1 1
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 1]
ww endturn

# --- Player 1 (Computer (A)) ---
ww build t:3,6 5
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 5]
ww endturn

# ============================================================
# Round 12
# ============================================================

# --- Player 0 (golilbattle) ---
ww move 4,3 5,4
ww assert exists unit 5,4
ww assert notexists unit 4,3
ww capture 5,4
ww move 3,3 4,5
ww assert exists unit 4,5
ww assert notexists unit 3,3
ww attack 4,5 3,6
ww assert unit 3,6 [health lte 8]
ww assert unit 4,5 [health lte 7]
ww move 4,2 3,3
ww assert exists unit 3,3
ww assert notexists unit 4,2
ww move 5,1 5,3
ww assert exists unit 5,3
ww assert notexists unit 5,1
ww move 2,4 2,5
ww assert exists unit 2,5
ww assert notexists unit 2,4
ww attack 1,5 3,6
ww assert unit 3,6 [health lte 8]
ww move 1,4 2,4
ww assert exists unit 2,4
ww assert notexists unit 1,4
ww move 4,1 4,3
ww assert exists unit 4,3
ww assert notexists unit 4,1
# Heal/Hold at 3,2
ww move 2,1 2,3
ww assert exists unit 2,3
ww assert notexists unit 2,1
ww move 0,3 1,4
ww assert exists unit 1,4
ww assert notexists unit 0,3
ww build t:2,1 2
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 2]
ww build t:0,3 2
ww assert exists unit 0,3
ww assert unit 0,3 [type eq 2]
# Heal/Hold at 3,5
ww attack 2,6 3,6
ww assert unit 3,6 [health lte 7]
ww assert unit 2,6 [health lte 7]
ww move 1,4 0,4
ww assert exists unit 0,4
ww assert notexists unit 1,4
ww endturn

# --- Player 1 (Computer (A)) ---
# Heal/Hold at 3,6
ww endturn

# ============================================================
# Round 13
# ============================================================

# --- Player 0 (golilbattle) ---
ww attack 3,5 3,6
ww assert unit 3,6 [health lte 8]
ww assert unit 3,5 [health lte 7]
ww attack 2,6 3,6
ww assert unit 2,6 [health lte 9]
ww attack 4,5 3,6
ww assert unit 3,6 [health lte 8]
ww move 2,5 2,6
ww assert exists unit 2,6
ww assert notexists unit 2,5
ww attack 2,6 3,6
ww assert unit 3,6 [health lte 2]
ww move 4,3 3,5
ww assert exists unit 3,5
ww assert notexists unit 4,3
ww move 2,3 2,5
ww assert exists unit 2,5
ww assert notexists unit 2,3
ww move 0,4 1,6
ww assert exists unit 1,6
ww assert notexists unit 0,4
ww move 3,3 4,4
ww assert exists unit 4,4
ww assert notexists unit 3,3
ww move 3,2 3,3
ww assert exists unit 3,3
ww assert notexists unit 3,2
ww move 2,1 1,3
ww assert exists unit 1,3
ww assert notexists unit 2,1
ww move 0,3 1,4
ww assert exists unit 1,4
ww assert notexists unit 0,3
ww move 5,3 5,5
ww assert exists unit 5,5
ww assert notexists unit 5,3
# Heal/Hold at 1,5
ww move 1,6 0,6
ww assert exists unit 0,6
ww assert notexists unit 1,6
ww move 2,4 1,6
ww assert exists unit 1,6
ww assert notexists unit 2,4
ww build t:2,1 8
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 8]
ww build t:0,3 7
ww assert exists unit 0,3
ww assert unit 0,3 [type eq 7]
ww endturn

# --- Player 1 (Computer (A)) ---
ww build t:3,6 5
ww assert exists unit 3,6
ww assert unit 3,6 [type eq 5]
ww endturn

# ============================================================
# Round 14
# ============================================================

# --- Player 0 (golilbattle) ---
ww build t:5,4 8
ww assert exists unit 5,4
ww assert unit 5,4 [type eq 8]
ww attack 5,5 3,6
ww assert unit 3,6 [health lte 9]
ww attack 4,5 3,6
ww assert unit 3,6 [health lte 9]
ww assert unit 4,5 [health lte 7]
ww attack 4,4 3,6
ww assert unit 3,6 [health lte 6]
ww attack 1,6 3,6
ww assert unit 3,6 [health lte 5]
ww move 3,5 3,6
ww assert exists unit 3,6
ww assert notexists unit 3,5
ww capture 3,6
# Heal/Hold at 3,3
# Heal/Hold at 1,5
ww move 2,1 4,2
ww assert exists unit 4,2
ww assert notexists unit 2,1
ww move 0,3 2,4
ww assert exists unit 2,4
ww assert notexists unit 0,3
ww move 1,3 2,3
ww assert exists unit 2,3
ww assert notexists unit 1,3
ww move 1,4 0,5
ww assert exists unit 0,5
ww assert notexists unit 1,4
ww move 0,6 3,5
ww assert exists unit 3,5
ww assert notexists unit 0,6
ww move 3,5 4,5
ww assert exists unit 4,5
ww assert notexists unit 3,5
ww move 2,5 3,5
ww assert exists unit 3,5
ww assert notexists unit 2,5
ww move 2,6 2,7
ww assert exists unit 2,7
ww assert notexists unit 2,6
ww build t:2,1 8
ww assert exists unit 2,1
ww assert unit 2,1 [type eq 8]
ww build t:0,3 1
ww assert exists unit 0,3
ww assert unit 0,3 [type eq 1]
ww endturn

# --- Player 1 (Computer (A)) ---
# No actions
ww endturn

# ============================================================
# Final Board State (from Lilbattle coordinate section)
# ============================================================

# Lilbattle (col,row) -> Our (Q,R) mapping:
# Lilbattle (7,5)    -> Q,R=(  5,  5) : Unit type 8:Artillery (Basic), golilbattle
# Lilbattle (2,1)    -> Q,R=(  2,  1) : Unit type 8:Artillery (Basic), golilbattle
# Lilbattle (7,4)    -> Q,R=(  5,  4) : Unit type 8:Artillery (Basic), golilbattle
# Lilbattle (6,6)    -> Q,R=(  3,  6) : Unit type 11:Capturing, golilbattle
# Lilbattle (6,4)    -> Q,R=(  4,  4) : Unit type 8:Artillery (Basic), golilbattle
# Lilbattle (1,3)    -> Q,R=(  0,  3) : Unit type 1:Soldier (Basic), golilbattle
# Lilbattle (3,5)    -> Q,R=(  1,  5) : Unit type 8:Artillery (Basic), golilbattle
# Lilbattle (4,6)    -> Q,R=(  1,  6) : Unit type 8:Artillery (Basic), golilbattle
# Lilbattle (4,3)    -> Q,R=(  3,  3) : Unit type 2:Soldier (Advanced), golilbattle
# Lilbattle (6,5)    -> Q,R=(  4,  5) : Unit type 3:Tank (Basic), golilbattle
# Lilbattle (5,2)    -> Q,R=(  4,  2) : Unit type 8:Artillery (Basic), golilbattle
# Lilbattle (4,4)    -> Q,R=(  2,  4) : Unit type 7:Hovercraft, golilbattle
# Lilbattle (3,3)    -> Q,R=(  2,  3) : Unit type 2:Soldier (Advanced), golilbattle
# Lilbattle (2,5)    -> Q,R=(  0,  5) : Unit type 2:Soldier (Advanced), golilbattle
# Lilbattle (5,5)    -> Q,R=(  3,  5) : Unit type 1:Soldier (Basic), golilbattle
# Lilbattle (5,7)    -> Q,R=(  2,  7) : Unit type 2:Soldier (Advanced), golilbattle

# Final board state assertions:
ww assert unit 5,5 [type eq 8, player eq 1, health eq 2]
ww assert unit 2,1 [type eq 8, player eq 1, health eq 10]
ww assert unit 5,4 [type eq 8, player eq 1, health eq 10]
ww assert unit 3,6 [type eq 11, player eq 1, health eq 8]
ww assert unit 4,4 [type eq 8, player eq 1, health eq 10]
ww assert unit 0,3 [type eq 1, player eq 1, health eq 10]
ww assert unit 1,5 [type eq 8, player eq 1, health eq 7]
ww assert unit 1,6 [type eq 8, player eq 1, health eq 10]
ww assert unit 3,3 [type eq 2, player eq 1, health eq 10]
ww assert unit 4,5 [type eq 3, player eq 1, health eq 10]
ww assert unit 4,2 [type eq 8, player eq 1, health eq 10]
ww assert unit 2,4 [type eq 7, player eq 1, health eq 10]
ww assert unit 2,3 [type eq 2, player eq 1, health eq 10]
ww assert unit 0,5 [type eq 2, player eq 1, health eq 10]
ww assert unit 3,5 [type eq 1, player eq 1, health eq 10]
ww assert unit 2,7 [type eq 2, player eq 1, health eq 7]

echo 'All assertions passed!'
