# vision_bot (planned)

This module is reserved for a future "screen vision + click" integration.

Goals (future):
- Capture the game window frames (screen scraping).
- Perceive visible state: hand tiles, discards, melds, action buttons (hu/peng/gang/pass), phase hints.
- Track state across frames and produce an information-set observation.
- Invoke the policy (same interface as training) to choose an action.
- Execute the action via mouse/keyboard events.

Important constraints:
- This repo only plans a "visible information + normal input" approach.
- Do not implement memory reading, code injection, or bypassing anti-cheat.
- Only integrate with platforms you are authorized to use.

Planned profile-driven configuration:
- Window matchers (title/process)
- Regions of interest (hand/discards/melds/buttons)
- Tile slot geometry / scaling
- Button templates and expected latencies
- Safety thresholds (confidence, retry limits, pause-on-uncertainty)
