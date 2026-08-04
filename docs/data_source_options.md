# Data Source Options

This note combines the two project PDFs with current official source positioning.

## Source Priority

### Tier 1: Foundation Data

Use these first because they support the team-level gold dataset without forcing player-level identity resolution on day one.

- **Existing BetBoard soccer lake**
  - Current source of historical `features`, `targets`, and `training` Parquet files.
  - Used to bootstrap `gold/prematch_model_input`.
- **Football-Data.co.uk**
  - Historical results and bookmaker odds.
  - Best fit for repeatable match-level backfills.
- **football-data.org**
  - Fixtures, competitions, teams, standings, squads, and match metadata.
  - Good fit for current-season refreshes and identity mapping.
  - Official docs: https://www.football-data.org/documentation/api
- **Open-Meteo**
  - Historical and forecast weather.
  - Good fit for venue-level match context.
  - Official site: https://open-meteo.com/

### Tier 2: Player, Lineup, Injury, And Event Enrichment

Use these after the team-level gold dataset is stable.

- **API-Football / API-Sports**
  - Candidate for player data, lineups, injuries, suspensions, events, and match/player statistics.
  - Good fit for operational enrichment if cost and coverage are acceptable.
  - Official docs: https://www.api-football.com/documentation-v3
- **Sportmonks Football API**
  - Candidate alternative for live scores, statistics, lineups, expected lineups, xG, odds, injuries/suspensions, and player statistics.
  - Strong candidate if expected lineups and injury/suspension coverage are important.
  - Official football API: https://www.sportmonks.com/football-api/
- **StatsBomb Open Data**
  - Best experimental source for high-granularity event data and player-event features.
  - Not a broad operational fixture source; coverage is limited to open-data competitions.
  - Official repository: https://github.com/hudl/open-data

## Implementation Order

1. Build `gold/prematch_model_input` from the existing historical lake.
2. Add Football-Data.co.uk collector for historical results and odds.
3. Add football-data.org collector for fixtures, teams, competitions, standings, and squads.
4. Add Open-Meteo weather enrichment.
5. Evaluate API-Football versus Sportmonks with a small paid/free trial matrix:
   - coverage for our priority leagues
   - historical player stats availability
   - expected lineup timing
   - confirmed lineup timing
   - injury/suspension update timestamp quality
   - rate limits and monthly cost
   - source IDs stable enough for player identity mapping
6. Add StatsBomb Open Data as an experimental event-data module, separate from production league coverage.

## Leakage Rules For Player Data

- Every lineup, injury, suspension, and player-stat row must carry `information_timestamp`.
- Expected lineups must be labeled separately from confirmed lineups.
- Confirmed lineups cannot be used for predictions whose cutoff is before lineup publication.
- Injury updates after kickoff cannot be used in prematch features.
- Player rolling features must only use matches before the prediction kickoff.
