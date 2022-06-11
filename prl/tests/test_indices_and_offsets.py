"""Rolling indices between frontend, and backend needs care.
The prl_environment.steinberger.PokerRL-environment,
 has the observation and seat indices relative to the BTN.
The vectorized observation is always relative to the current player.
The frontend needs the seat indices relative to the HERO seat.

Additionally, when a players stack is 0 at the beginning of a round, he or she is eliminated.
His seat is kept in the frontend, and thus seat indices and their order does not change in the frontend.
However, the prl_environment.steinberger.PokerRL-environment, needs to be reset
with a decreased number of players, because it does not work,
if it is reset with a person that has a stack size of 0.

Furthermore, we need to keep track of the BTN position. Lets look at an example

Round starts with 6 players and stacks. The button is randomly determined at
the very first reset, and gets seat_id=2. After that the BTN goes to the next
 player on the left.
These are the starting stacks of round=1 as seen by the frontend.
{
    200,  # HERO
    140,
    200,  # BTN
    200,
    200
    200
}
During the game, 3 players go all in and we get final stacks
{
    0,    # HERO
    140,
    800,  # BTN
    0,
    0
    200
}

Now the button must be propagated to seat_id=5.
On end of a round that is signalled by the api via a `done`-flag response,
the frontend sends a POST request to the /reset endpoint of the API using the final stack sizes.

It is now the API`s job -upon receiving the reset-request, to translate
{
    0,    # HERO
    140,
    800,  # BTN
    0,
    0
    200
}

to
{
    200,  # BTN propagated to the left
    140,
    800,
} for the backend to reset the environment
 - using the new BTN position
 - and without eliminated players,
and translate it back for the frontend to
{
    0,    # HERO
    140,
    800,
    0,
    0
    200  # BTN
}
"""