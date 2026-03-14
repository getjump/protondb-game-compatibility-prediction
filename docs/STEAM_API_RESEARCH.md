# Steam API Research: New Signals for Proton Compatibility Prediction

> Date: 2026-03-14
>
> Goal: identify Steam API endpoints (beyond Steam Store web API) that provide
> data useful for predicting game compatibility with Proton/Wine on Linux.
> We already have: ProtonDB reports, Steam Store appdetails, PCGamingWiki,
> Deck Verified status, ProtonDB contributor data.

---

## Table of Contents

1. [ISteamApps](#1-isteamapps)
2. [IPlayerService](#2-iplayerservice)
3. [ISteamUserStats](#3-isteamuserstats)
4. [Steam Reviews API](#4-steam-reviews-api)
5. [Steam Store API (appdetails)](#5-steam-store-api-appdetails)
6. [ISteamNews](#6-isteamnews)
7. [Steam Deck Compatibility Report API](#7-steam-deck-compatibility-report-api)
8. [IStoreService](#8-istoreservice)
9. [Steam Store Package / DLC APIs](#9-steam-store-package--dlc-apis)
10. [Inventory / Economy APIs](#10-inventory--economy-apis)
11. [Undocumented / Hard-to-Access Data](#11-undocumented--hard-to-access-data)
12. [Summary: Recommended New Data Sources](#12-summary-recommended-new-data-sources)

---

## 1. ISteamApps

Base URL: `https://api.steampowered.com/ISteamApps/`

### Public (No API Key)

#### GetServersAtAddress

```
GET /ISteamApps/GetServersAtAddress/v1/?addr=<IP>
```

- **Key required:** No
- **Returns:** list of game servers at a given IP address.
- **Usefulness: NONE.** Server infrastructure data, irrelevant for compatibility.

#### UpToDateCheck

```
GET /ISteamApps/UpToDateCheck/v1/?appid=<id>&version=<ver>
```

- **Key required:** No
- **Returns:** `up_to_date` (bool), `version_is_listable` (bool),
  `required_version` (uint), `message` (string).
- **Usefulness: LOW.** Could indicate how actively a game is maintained, but
  requires knowing the currently installed version number in advance.

#### GetSDRConfig

```
GET /ISteamApps/GetSDRConfig/v1/?appid=<id>
```

- **Key required:** No
- **Returns:** Steam Datagram Relay network config: relay PoP locations,
  IPs, port ranges, latency tables, certificates.
- **Usefulness: LOW.** The *existence* of an SDR config is a weak binary
  feature: the game uses Valve's networking stack, which may correlate with
  better Proton support. Not worth the call volume.

### Publisher-Only (Require Publisher API Key)

These endpoints are inaccessible without a publisher key tied to the
specific app. Listed for completeness:

| Method | What It Does |
|--------|--------------|
| `GetAppBetas/v1` | Beta branches for an app |
| `GetAppBuilds/v1` | Build history |
| `GetAppDepotVersions/v1` | Depot version info |
| `GetPlayersBanned/v1` | Banned player data |
| `GetServerList/v1` | Dedicated server list |
| `GetPartnerAppListForWebAPIKey/v2` | Apps owned by a publisher key |
| `SetAppBuildLive/v2` | Activate a build (POST) |

**Verdict: ISteamApps provides almost nothing useful publicly.** The
`GetAppList` method is deprecated; Valve directs users to
`IStoreService/GetAppList` instead.

---

## 2. IPlayerService

Base URL: `https://api.steampowered.com/IPlayerService/`

All methods require an API key (regular Steam Web API key, not publisher).

#### GetOwnedGames

```
GET /IPlayerService/GetOwnedGames/v1/?key=<key>&steamid=<id>&include_appinfo=true&include_played_free_games=true
```

- **Key required:** Yes (regular)
- **Returns per game:** `appid`, `name`, `playtime_forever` (minutes),
  `playtime_2weeks`, `img_icon_url`, `has_community_visible_stats`.
- **CRITICAL FINDING: No per-platform playtime.** Fields like
  `playtime_linux`, `playtime_windows`, `playtime_mac`, `playtime_deck`
  do NOT exist in the response. The Steam client tracks this internally
  but the API does not expose it.
- **Usefulness: MEDIUM.** Could cross-reference a ProtonDB reporter's total
  library/playtime as an "experience" feature, but this requires iterating
  over individual profiles. Not scalable for 350K+ reports.

#### GetRecentlyPlayedGames

```
GET /IPlayerService/GetRecentlyPlayedGames/v1/?key=<key>&steamid=<id>&count=<n>
```

- **Key required:** Yes (regular)
- **Returns:** Same fields as GetOwnedGames, limited to recently played.
- **Usefulness: LOW.** Same limitations as above.

#### GetSingleGamePlaytime

```
GET /IPlayerService/GetSingleGamePlaytime/v1/?key=<key>&steamid=<id>&appid=<id>
```

- **Key required:** Yes (partner only via `partner.steam-api.com`)
- **Usefulness: NONE for us.** Partner-restricted.

#### GetSteamLevel / GetBadges / GetCommunityBadgeProgress

- User profile metadata (level, badges, badge progress).
- **Usefulness: NONE.** Not related to game compatibility.

**Verdict: IPlayerService is not useful for our task.** No platform-specific
data, and we already have per-report playtime from ProtonDB.

---

## 3. ISteamUserStats

Base URL: `https://api.steampowered.com/ISteamUserStats/`

### No API Key Required

#### GetNumberOfCurrentPlayers

```
GET /ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid=<id>
```

- **Key required:** No
- **Returns:**
  ```json
  { "response": { "player_count": 832038, "result": 1 } }
  ```
- **Rate limits:** Unknown explicit limit, but no key required so likely
  generous. Can be polled for all ~50K games.
- **Usefulness: MEDIUM-HIGH.** Current player count is a proxy for popularity,
  which correlates with:
  - More Proton testing/fixes by community
  - More pressure on devs to address Linux issues
  - Better wine/proton compatibility coverage
  - Larger sample size in ProtonDB
- **Feature ideas:**
  - `log_current_players` — log-scaled player count
  - `is_active_game` — binary: `current_players > 0`
  - `popularity_bucket` — binned into tiers

#### GetGlobalAchievementPercentagesForApp

```
GET /ISteamUserStats/GetGlobalAchievementPercentagesForApp/v2/?gameid=<id>
```

- **Key required:** No
- **Returns:** Array of `{ "name": "ACH_ID", "percent": 75.4 }` for every
  achievement in the game.
- **Usefulness: HIGH.** Achievement completion rates are a proxy for game
  stability and playability:
  - Very low first-achievement completion (< 50%) may indicate
    crashes/launch issues across all platforms
  - Median achievement % indicates how "completable" the game is
  - The *number* of achievements is itself a feature (games with
    achievements tend to be more mature/polished)
  - Achievement drop-off curve shape could indicate technical issues
    vs. difficulty
- **Feature ideas:**
  - `achievement_count` — total number of achievements
  - `median_achievement_pct` — median completion rate
  - `min_achievement_pct` — rarest achievement rate
  - `first_achievement_pct` — highest completion rate (proxy for
    "can players even start and play the game?")
  - `achievement_completion_variance` — spread in completion rates
  - `achievement_entropy` — information-theoretic measure of distribution

### API Key Required (Regular)

#### GetSchemaForGame

```
GET /ISteamUserStats/GetSchemaForGame/v2/?key=<key>&appid=<id>&l=english
```

- **Key required:** Yes (regular)
- **Returns:** Game name + full list of stat/achievement definitions
  (internal names, display names, descriptions, icon URLs).
- **Usefulness: LOW.** The actual completion percentages (above) are more
  useful than the schema metadata.

#### GetGlobalStatsForGame

```
GET /ISteamUserStats/GetGlobalStatsForGame/v1/?key=<key>&appid=<id>&count=1&name[0]=<stat_name>
```

- **Key required:** Yes (regular)
- **Returns:** Aggregated stat values for named stats. Supports date ranges.
- **Limitation:** Requires knowing stat names in advance. Developer must
  mark stats as globally aggregated. Many games don't do this.
- **Usefulness: LOW.** Too game-specific, not generalizable.

#### GetPlayerAchievements / GetUserStatsForGame

- Per-user achievement/stat data. Requires user's steamid.
- **Usefulness: NONE at scale.**

**Verdict: `GetNumberOfCurrentPlayers` and
`GetGlobalAchievementPercentagesForApp` are the best endpoints here.**
Both require no API key and return actionable signals.

---

## 4. Steam Reviews API

### Endpoint

```
GET https://store.steampowered.com/appreviews/<appid>?json=1
```

No authentication required (it is a Store web endpoint).

### Parameters

| Parameter | Values | Notes |
|-----------|--------|-------|
| `filter` | `recent`, `updated`, `all` | Sort order |
| `language` | ISO code or `all` | |
| `day_range` | 1-365 | Only with `filter=all` |
| `review_type` | `all`, `positive`, `negative` | |
| `purchase_type` | `all`, `steam`, `non_steam_purchase` | |
| `num_per_page` | 1-100 | |
| `cursor` | pagination token | Use `*` initially |
| `filter_offtopic_activity` | 0 or 1 | Exclude review bombs |

### Response Structure

**Query summary (aggregate, returned on first page):**

| Field | Example | Notes |
|-------|---------|-------|
| `review_score` | 8 | 1-9 scale |
| `review_score_desc` | "Very Positive" | Human-readable |
| `total_positive` | 1,040,997 | |
| `total_negative` | 78,171 | |
| `total_reviews` | 1,119,168 | |

**Per review:**

| Field | Type | Notes |
|-------|------|-------|
| `voted_up` | bool | Positive or negative review |
| `votes_up` | int | Helpfulness votes |
| `votes_funny` | int | |
| `weighted_vote_score` | float | Steam's helpfulness weight |
| `steam_purchase` | bool | Purchased on Steam vs. key |
| `received_for_free` | bool | |
| `written_during_early_access` | bool | |
| `primarily_steam_deck` | bool | **Reviewer primarily played on Deck** |
| `author.playtime_forever` | int | Minutes |
| `author.playtime_at_review` | int | Minutes at time of review |
| `author.playtime_last_two_weeks` | int | |
| `author.num_games_owned` | int | |
| `author.num_reviews` | int | |
| `language` | string | ISO code |
| `timestamp_created` | int | Unix timestamp |
| `review` | string | Full review text |

### Key Findings

- **No OS filter parameter.** You cannot filter reviews by Linux/Windows/Mac.
  The only platform signal is `primarily_steam_deck` per review.
- **Rate limits:** Similar to appdetails — approximately 200 requests per
  5 minutes on `store.steampowered.com`.

### Usefulness: HIGH

| Feature Idea | How to Get It | Cost |
|-------------|---------------|------|
| `review_score` (1-9) | query_summary on first page | 1 call/game |
| `total_reviews` | query_summary | 1 call/game |
| `positive_ratio` | `total_positive / total_reviews` | 1 call/game |
| `deck_review_ratio` | Paginate all reviews, count `primarily_steam_deck=true` | 10-100+ calls/game |
| `early_access_review_ratio` | Ratio of `written_during_early_access` reviews | Pagination needed |
| Review text mining | Search for "linux", "proton", "crash", "steam deck" | NLP, expensive |

The query_summary fields (score, totals, ratio) are obtainable with a
single API call per game and provide genuine signal: overall game quality
correlates with developer investment in compatibility.

---

## 5. Steam Store API (appdetails)

### Endpoint

```
GET https://store.steampowered.com/api/appdetails?appids=<id>
```

Optional: `&filters=<comma-separated>` (e.g. `categories,platforms`)

### Rate Limits

~200 requests per 5 minutes. Multiple appids no longer work reliably
without `filters=price_overview`.

### All Useful Fields for Compatibility Prediction

| Field | Example Value | ML Feature Potential |
|-------|---------------|---------------------|
| `platforms.windows` | `true` | Baseline |
| `platforms.mac` | `true`/`false` | Mac support correlates with cross-platform effort |
| `platforms.linux` | `true`/`false` | **HIGH** — native Linux build = likely works |
| `categories[]` | `[{id, description}]` | Controller support, multiplayer, VAC, Steam Cloud |
| `genres[]` | `[{id, description}]` | Genre correlates with engine/technology choices |
| `pc_requirements` | HTML string | **VERY HIGH** — parseable for DX version, 64-bit, GPU |
| `linux_requirements` | HTML string | Existence = native build; content reveals Vulkan/OpenGL |
| `mac_requirements` | HTML string | Cross-platform indicator |
| `controller_support` | `"full"` / `"partial"` / null | Full controller = better Deck compat |
| `required_age` | `0`-`18` | Mature games tend to be AAA (different tech profile) |
| `is_free` | `true`/`false` | F2P games have different compat patterns |
| `developers[]` | `["FromSoftware"]` | Developer track record as a feature |
| `publishers[]` | `["Bandai Namco"]` | Publisher track record |
| `release_date.date` | `"Feb 24, 2022"` | Game age; older = more time for Proton fixes |
| `metacritic.score` | `94` | Game quality / dev investment proxy |
| `recommendations.total` | `807782` | Popularity proxy |
| `achievements.total` | `42` | Game maturity indicator |
| `dlc[]` | `[2778590, ...]` | DLC count = ongoing support |
| `type` | `"game"` | Filter out non-games |
| `supported_languages` | HTML string | Localization breadth = dev investment |

### Parsing pc_requirements for DirectX Version

The `pc_requirements.minimum` and `pc_requirements.recommended` fields
contain HTML with parseable technical data. Key extraction targets:

| Signal | How It Appears in HTML | Feature |
|--------|------------------------|---------|
| DirectX version | "DirectX: Version 12", "DirectX 11" | `dx_version` (9/10/11/12) |
| 64-bit required | "64-bit processor", "Requires a 64 Bit" | `requires_64bit` (bool) |
| Vulkan | "Vulkan" in linux_requirements | `uses_vulkan` (bool) |
| OpenGL | "OpenGL 4.5" | `uses_opengl` (bool) |
| Min RAM | "8 GB RAM" | `min_ram_gb` (int) |
| Min VRAM | "4 GB VRAM" | `min_vram_gb` (int) |
| Storage | "60 GB available space" | `storage_gb` (int) |

DirectX version is one of the strongest signals for Proton compatibility:
- **DX9/DX10** — generally excellent Proton compat (DXVK mature)
- **DX11** — usually works well (DXVK)
- **DX12** — more challenging (VKD3D-proton, still improving)
- **Vulkan native** — excellent compat, runs natively

### Category IDs of Interest

Selected category IDs observed across multiple games:

| Category ID | Description | Relevance |
|-------------|-------------|-----------|
| 2 | Single-player | Game type |
| 1 | Multi-player | Multiplayer may use anti-cheat |
| 9 | Co-op | |
| 28 | Full controller support | Deck compat |
| 18 | Partial controller support | |
| 8 | VAC enabled | Anti-cheat presence |
| 22 | Steam Achievements | Game maturity |
| 23 | Steam Cloud | |
| 29 | Steam Trading Cards | |
| 30 | Steam Workshop | |
| 42-44 | Remote Play variants | |

**Note:** There is NO category for "Linux/SteamOS support", "Steam Play",
or "Proton compatible" — platform support is only in the `platforms` object.

---

## 6. ISteamNews

### Endpoint

```
GET https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/?appid=<id>&count=<n>&maxlength=<len>
```

### Parameters

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `appid` | uint32 | Yes | Game ID |
| `count` | uint32 | No | Max items to return |
| `maxlength` | uint32 | No | Truncate content to N chars (0 = full) |
| `enddate` | uint32 | No | Unix timestamp upper bound |
| `feeds` | string | No | Comma-separated feed names |
| `tags` | string | No | Filter by tag (e.g. `patchnotes`) |

- **Key required:** No
- **Rate limits:** Unknown, likely similar to other Steam API endpoints.

### Response Per News Item

| Field | Type | Notes |
|-------|------|-------|
| `gid` | string | Unique news item ID |
| `title` | string | Headline |
| `url` | string | Full URL |
| `author` | string | |
| `contents` | string | Text/HTML body |
| `feedlabel` | string | e.g. "Community Announcements" |
| `date` | int | Unix timestamp |
| `feedname` | string | Internal feed identifier |
| `feed_type` | int | Feed type enum |
| `tags[]` | array | e.g. `["patchnotes"]` |

### Usefulness: MEDIUM

| Feature Idea | Description | Cost |
|-------------|-------------|------|
| `total_news_count` | Volume of news/updates | 1 call/game |
| `patchnotes_count` | Count items with `patchnotes` tag | 1 call/game (with `tags=patchnotes`) |
| `days_since_last_news` | Recency of latest update | 1 call/game |
| `days_since_last_patch` | Recency of latest patchnotes | 1 call/game |
| `has_proton_mention` | Text search in news for "linux"/"proton"/"vulkan" | NLP on content |
| `patch_frequency_12mo` | Patches per year | 1 call/game |

Actively patched games are more likely to have received Proton fixes.
Recency of last update indicates whether developers are still engaged.

**Caveat:** Text mining news content for Proton mentions is expensive
(requires fetching full content for many items) and noisy (many false
positives from community announcements).

---

## 7. Steam Deck Compatibility Report API

### Endpoint

```
GET https://store.steampowered.com/saleaction/ajaxgetdeckappcompatibilityreport?nAppID=<id>
```

- **Key required:** No (unauthenticated Store endpoint)
- **Rate limits:** Likely same as Store (~200 req/5 min)

### Response Structure

```json
{
  "success": 1,
  "results": {
    "appid": 1245620,
    "resolved_category": 3,
    "resolved_items": [
      {
        "display_type": 4,
        "loc_token": "#SteamDeckVerified_TestResult_DefaultControllerConfigFullyFunctional"
      },
      {
        "display_type": 4,
        "loc_token": "#SteamDeckVerified_TestResult_ControllerGlyphsMatchDeckDevice"
      },
      {
        "display_type": 4,
        "loc_token": "#SteamDeckVerified_TestResult_InterfaceTextIsLegible"
      },
      {
        "display_type": 4,
        "loc_token": "#SteamDeckVerified_TestResult_DefaultConfigurationIsPerformant"
      }
    ],
    "steam_deck_blog_url": "",
    "search_id": null,
    "steamos_resolved_category": 2,
    "steamos_resolved_items": [
      {
        "display_type": 3,
        "loc_token": "#SteamOS_TestResult_GameStartupFunctional"
      }
    ]
  }
}
```

### Category Values

| `resolved_category` | Meaning |
|----------------------|---------|
| 0 | Unknown (not tested) |
| 1 | Unsupported |
| 2 | Playable |
| 3 | Verified |

Same values apply to `steamos_resolved_category`.

### display_type Values

| Value | Meaning | Visual |
|-------|---------|--------|
| 1 | Informational / limitation | Yellow note |
| 2 | Failure / unsupported | Red X |
| 3 | Warning / works with caveats | Yellow warning |
| 4 | Pass | Green checkmark |

### Known loc_tokens (Test IDs)

**Steam Deck tests:**
- `#SteamDeckVerified_TestResult_DefaultControllerConfigFullyFunctional`
- `#SteamDeckVerified_TestResult_ControllerGlyphsMatchDeckDevice`
- `#SteamDeckVerified_TestResult_ControllerGlyphsDoNotMatchDeckDevice`
- `#SteamDeckVerified_TestResult_InterfaceTextIsLegible`
- `#SteamDeckVerified_TestResult_DefaultConfigurationIsPerformant`
- `#SteamDeckVerified_TestResult_ExternalControllersNotSupportedPrimaryPlayer`
- `#SteamDeckVerified_TestResult_UnsupportedAntiCheat_Other`

**SteamOS tests:**
- `#SteamOS_TestResult_GameStartupFunctional`
- `#SteamOS_TestResult_ExternalControllersNotSupportedPrimaryPlayer`

### Usefulness: VERY HIGH

This endpoint provides data beyond what a simple Verified/Playable/Unsupported
enum captures:

1. **Granular test results** — not just the overall category but *why* a game
   is Playable vs. Verified (anti-cheat failure? controller glyphs? text
   legibility? performance?). Each test is a separate binary feature.
2. **`steamos_resolved_category`** — a SEPARATE category from Steam Deck.
   A game can be Deck Verified (3) but SteamOS Playable (2). This dual
   system provides additional nuance.
3. **Anti-cheat detection** — `UnsupportedAntiCheat_Other` directly flags
   anti-cheat as the blocker.

**Feature ideas:**
- `deck_category` (0-3)
- `steamos_category` (0-3) — **new vs. what we have**
- `deck_controller_pass` (bool)
- `deck_glyphs_pass` (bool)
- `deck_text_legible` (bool)
- `deck_performance_pass` (bool)
- `deck_anticheat_fail` (bool)
- `deck_test_pass_count` (int)
- `deck_test_fail_count` (int)

---

## 8. IStoreService

### GetAppList

```
GET https://partner.steam-api.com/IStoreService/GetAppList/v1/?key=<key>&max_results=<n>&include_games=true
```

- **Key required:** Yes (partner)
- **Returns per app:** `appid`, `last_modified`, `price_change_number`
- **Usefulness: LOW.** Just a paginated app list with timestamps.
  Less data than `appdetails`. Meant for bulk app enumeration.

**Verdict: IStoreService is not useful for our task.**

---

## 9. Steam Store Package / DLC APIs

### Package Details

```
GET https://store.steampowered.com/api/packagedetails?packageids=<id>
```

Returns package metadata: apps included, platforms, pricing.
**Usefulness: LOW** — app-level data is more relevant.

### DLC List

```
GET https://store.steampowered.com/api/dlcforapp?appid=<id>
```

Returns DLC list with per-DLC `platforms` (win/mac/linux),
`controller_support`, `price_overview`.
**Usefulness: LOW** — DLC platform support generally mirrors base game.

---

## 10. Inventory / Economy APIs

| Interface | Purpose | Usefulness |
|-----------|---------|------------|
| IEconService | Trade offers, trade history | NONE |
| IEconMarketService | Market listings, prices | NONE |
| ISteamEconomy | In-game item data | NONE |

**Verdict: Zero relevance to compatibility prediction.**

---

## 11. Undocumented / Hard-to-Access Data

### What SteamDB Has But We Cannot Get Via Public Web API

SteamDB accesses Steam's internal app configuration via **PICS** (Package
Info Cache System), which is part of the Steam client protocol (CM —
Connection Manager), NOT the Web API. PICS provides:

| Data | Description | Compatibility Signal |
|------|-------------|----------------------|
| Launch configurations | Executables, arguments, per-OS launch options | OS-specific exe = native build |
| Depot information | Separate depots per OS | Linux depot = native port |
| Technology tags | Engine, graphics API, anti-cheat, DRM | Direct compat indicators |
| App type flags | Game, tool, demo, DLC, etc. | Filter non-games |
| 32/64-bit flags | Architecture of executables | 64-bit = modern, better compat |
| OS-specific configs | Windows-only vs. cross-platform launch entries | Platform support detail |

### SteamCMD Approach

```
steamcmd +login anonymous +app_info_print <appid> +quit
```

This dumps app configuration in VDF (Valve Data Format) including depot
info and launch options. Technically feasible but:
- Requires SteamCMD installed
- Rate-limited by Steam login session
- VDF parsing needed (not JSON)
- One call per app (not bulk)

**This is the richest data source we cannot easily access.** SteamDB and
similar projects use this internally.

### Data Confirmed NOT Available via Any Public API

| Data | Status |
|------|--------|
| Per-platform playtime (`playtime_linux`, `playtime_windows`, etc.) | Tracked by Steam client, NOT exposed via API |
| Game executable details (PE header, linked DLLs, 32/64-bit) | Not available |
| DirectX/Vulkan runtime detection | Only inferrable from requirements text |
| Anti-cheat type (specific product) | Only via Deck compat report or PCGamingWiki |
| Proton compatibility whitelists | Valve internal, not public |
| Wine/Proton regression data | Not exposed |
| Steam Play game-specific configs | Only via PICS/SteamCMD |

---

## 12. Summary: Recommended New Data Sources

### Tier 1 — High Value, Easy to Collect (No API Key, 1 Call per Game)

| Endpoint | Feature Ideas | Key? | Calls |
|----------|---------------|------|-------|
| `GetNumberOfCurrentPlayers` | `log_current_players`, `is_active_game`, `popularity_bucket` | No | 1/game |
| `GetGlobalAchievementPercentagesForApp` | `achievement_count`, `median_ach_pct`, `min_ach_pct`, `first_ach_pct`, `ach_variance` | No | 1/game |
| Deck Compatibility Report | `deck_category`, `steamos_category`, per-test binary features, `has_anticheat_fail`, `test_pass_count` | No | 1/game |
| Reviews query_summary | `review_score` (1-9), `total_reviews`, `positive_ratio` | No | 1/game |
| `appdetails` pc_requirements parsing | `dx_version` (9/10/11/12), `requires_64bit`, `min_ram_gb`, `uses_vulkan` | No | Already collected |

**Estimated total calls for Tier 1:** ~200K (4 endpoints x ~50K games).
At 200 req/5 min for Store endpoints and uncapped for API endpoints,
this is feasible over a few hours.

### Tier 2 — Medium Value, Moderate Cost

| Endpoint | Feature Ideas | Calls |
|----------|---------------|-------|
| Reviews `primarily_steam_deck` ratio | `deck_review_ratio` | 10-100/game (pagination) |
| ISteamNews patch frequency | `patchnotes_count_12mo`, `days_since_last_update`, `patch_frequency` | 1/game |
| ISteamNews Proton mentions | `has_proton_mention_in_news` | NLP on content |
| `appdetails` category parsing | `has_full_controller_support`, `has_steam_cloud`, `is_multiplayer`, `has_vac` | Already collected |
| `appdetails` developer/publisher | Developer Proton track record (aggregate per developer) | Already collected |

### Tier 3 — High Value But Hard to Access

| Source | Data | Difficulty |
|--------|------|------------|
| SteamCMD `app_info_print` | Linux depot exists, launch configs, OS flags, architecture | VDF parsing, slow, requires SteamCMD |
| `pc_requirements` HTML parsing | DirectX version, Vulkan mention, 64-bit flag | Regex/NLP, already planned in PLAN_LLM.md |
| Review text mining | "proton", "linux", "crash", "deck" keyword frequency | NLP, expensive pagination |

### What We Already Have vs. What's Genuinely New

**Already covered** (per existing enrichment plans):
- Steam Store data (genres, categories, release date, platforms)
- PCGamingWiki (engine, DRM, anti-cheat)
- Deck Verified status (category only)
- ProtonDB contributor data (report_tally, playtime)

**Genuinely new signals worth adding:**

1. **Achievement statistics** from `GetGlobalAchievementPercentagesForApp`
   — free, no key, 1 call/game, provides 5+ features
2. **Current player count** from `GetNumberOfCurrentPlayers`
   — free, no key, 1 call/game
3. **Deck compatibility granular test results** — individual pass/fail
   per test, beyond just the overall Verified/Playable/Unsupported category
4. **SteamOS resolved category** — separate from Deck category, available
   in same Deck compat API call (no extra cost)
5. **Review score and positive ratio** from reviews query_summary
   — 1 call/game
6. **Patch frequency** from ISteamNews — 1 call/game
7. **DirectX version** parsed from `pc_requirements` HTML
   — if not already being extracted

### Rate Limit Summary

| Domain | Limit | Notes |
|--------|-------|-------|
| `store.steampowered.com` | ~200 req / 5 min | appdetails, reviews, deck compat |
| `api.steampowered.com` | No documented limit (keyless) | GetNumberOfCurrentPlayers, achievements |
| `api.steampowered.com` (with key) | 100K calls/day typical | IPlayerService, GetSchemaForGame |
| `partner.steam-api.com` | Publisher-restricted | Most ISteamApps methods |

 Из PoC мы видели что PICS уже даёт review_score (1-9) и review_percentage (0-100) для 96% игр — бесплатно, в том же batch запросе. Это эквивалент query_summary из Reviews API, но без отдельных вызовов.

  Что Reviews API даёт сверх PICS:
  - primarily_steam_deck per review — ratio Deck юзеров. Но нужно 10-100 calls/game (пагинация) = ~300K-3M запросов
  - Текст отзывов — keyword mining ("proton", "linux", "crash"). Тоже дорого

  Стоит ли:
  - review_score + review_percentage — уже есть из PICS, бесплатно
  - Deck review ratio — интересно, но стоит ~300K API calls при rate limit 200/5min = ~12 часов. И сигнал косвенный
  - Review text mining — слишком дорого для marginal value

  Вердикт: нет. PICS уже покрывает review data. Отдельный Reviews API не стоит затрат — review_score из PICS + наши ProtonDB reports дают больше сигнала чем Steam reviews.