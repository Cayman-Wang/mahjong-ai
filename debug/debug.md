# 四川麻将强化学习引擎全量代码审查报告

**审查时间**: 2026年2月10日  
**审查者**: GitHub Copilot (Claude Sonnet 4.5)  
**仓库**: /home/grasp/Desktop/mahjong-ai  
**语言**: Python (src-layout: src/mahjong_ai)  
**玩法**: 四川麻将（血战到底）

---

## 1. 结论摘要

### 当前状态
- ✅ **核心引擎基本可运行**，单元测试覆盖关键场景（换三张守恒、响应多胡、零和结算）
- ✅ **架构清晰**：核心无依赖、可选 RL/vision 扩展、seed 可复现机制正常
- ✅ **代码质量**：类型注解完整、模块划分合理、无明显安全漏洞
- ⚠️ **存在 3 个 P0 阻塞性问题**（抢杠胡牌数守恒、一炮多响结算逻辑、流局死循环风险）
- ⚠️ **P1 边界问题较多**，但不会立即导致崩溃
- ⚠️ **测试覆盖严重不足**：仅 4 个单测，缺少不变量测试和边界场景

### 最重要的改动建议（按优先级）

1. **立即修复 P0-1**（抢杠胡牌数守恒）：会导致总牌数!=108，破坏游戏不变量
2. **立即修复 P0-2**（一炮多响结算逻辑）：当前点炮结算会导致零和破坏
3. **补充不变量测试**：目前仅 4 个单测，需至少 10+ 属性测试（牌数守恒、零和、mask-step 一致性）
4. **修复 P0-3**（流局死循环）：防御性编程，避免极端场景卡死

### 训练可用性判断

**现在是否可直接上 PPO 自对弈？**
- **理论上可以**：观测/奖励/mask/多智能体接口都已实现
- **但强烈建议修复 P0 后再开工**，原因：
  - P0-1（抢杠牌数）会让智能体学到错误策略
  - P0-2（一炮多响结算）会让智能体学到错误的价值函数
  - 测试覆盖不足，训练中容易遇到未发现的 bug

---

## 2. P0 问题（阻塞级 - 会导致规则错误/崩溃/不可复现/零和破坏）

### P0-1: 抢杠胡后牌数守恒被破坏 🔴

**严重性**: P0（规则错误 + 不变量破坏）  
**文件**: `src/mahjong_ai/core/engine.py:634-636`

#### 问题描述

在 `RESPONSE_QIANGGANG` 阶段，如果有玩家抢杠胡，引擎的处理流程：

```python
# L636: 先从 actor 手牌中扣除补杠牌
counts_remove(state.players[actor].hand, t, 1)

# L638-672: 对每个 winner 结算
for winner in hu:
    p = state.players[winner]
    tmp = p.hand.copy()
    tmp[t] += 1  # ⚠️ 仅在临时副本中加牌
    ctx = WinContext(..., rob_kong=True)
    fan = compute_fan(counts=tmp, player=p, ctx=ctx, rules=self.rules)
    # ... settle_hu ...
    state.players[winner].won = True
```

**问题核心**：
- 被抢的牌（t）从 actor 的 hand 中扣除了（L636）
- winner 在胡牌判定时只在 tmp 副本中加牌（L645），**winner 的真实 hand 没有获得这张牌**
- 结果：**该牌从游戏中消失**，总牌数 < 108

#### 触发条件
任何抢杠胡场景（补杠声明后被其他玩家抢杠胡）

#### 最小复现
```python
# 构造场景：
# - player0 有 5W 的碰，手里再摸到 5W 补杠
# - player1 听 5W，可以抢杠胡
state.players[0].melds = [Meld(MeldKind.PENG, tile_id(0,5), from_player=2)]
state.players[0].hand[tile_id(0,5)] = 1  # 补杠牌
state.players[1].hand = construct_ting_hand_waiting_5W()

# player0 补杠 5W -> 进入 RESPONSE_QIANGGANG
# player1 选择 HU
# 执行后：
#   - player0.hand[5W] 被扣除（-1）
#   - player1.hand[5W] 没有增加（仅 tmp 中+1）
#   - 5W 消失，sum(all_tiles) = 107
```

#### 影响面
- 破坏核心不变量（总牌数==108）
- 后续逻辑可能误判（牌库计数错误）
- 测试未覆盖此场景（test_core_response_multi_hu 没测抢杠）

#### 修复建议

**方案 A（推荐）**：抢杠胡时，将扣除的牌加入 winner 手牌

```python
# 在 RESPONSE_QIANGGANG 的 if hu: 分支中（约 L638-672）
if hu:
    if not self.rules.allow_yipao_duoxiang:
        winner = pick_closest(hu)
        hu = [winner] if winner is not None else []

    # ✅ 扣除 actor 的补杠牌
    if state.players[actor].hand[t] < 1:
        raise ValueError("actor does not have the tile for bu-gang")
    counts_remove(state.players[actor].hand, t, 1)

    # ✅ 方案A：将抢到的牌加入第一个 winner 的手牌（物理转移）
    primary_winner = hu[0] if hu else None
    if primary_winner is not None:
        counts_add(state.players[primary_winner].hand, t, 1)

    # 对每个 winner 结算（注意：若多人抢杠，只有第一人真实获得牌）
    for winner in hu:
        p = state.players[winner]
        tmp = p.hand.copy()  # 现在 primary_winner 的 tmp 已包含被抢的牌
        if winner != primary_winner:
            tmp[t] += 1  # 其他 winner 仅虚拟加牌判定
        ctx = WinContext(..., rob_kong=True)
        fan = compute_fan(counts=tmp, player=p, ctx=ctx, rules=self.rules)
        # ... settle_hu ...
        state.players[winner].won = True
    
    state.pending_kong = None
    # ... 后续逻辑 ...
```

**方案 B（替代）**：不扣 actor 的牌，仅在 tmp 中虚拟加牌

```python
# 不修改 actor.hand，仅在 winner 的 tmp 中+1
# 但需调整语义：抢杠胡时，actor 的补杠牌"冻结"不完成补杠，也不扣除
# （需相应修改 Meld 状态，逻辑较复杂，不推荐）
```

**单元测试**：
```python
def test_qianggang_tile_conservation(self):
    """验证抢杠胡后总牌数守恒"""
    engine = GameEngine(enable_events=False)
    state = engine.reset(seed=1)
    
    # 构造：player0 有 5W 的碰，摸到 5W 补杠
    # player1 听 5W
    # ... 构造状态 ...
    
    total_before = count_all_tiles(state)
    
    # player0 补杠 -> player1 抢杠胡
    res = engine.step(state, {1: encode_action(ActionKind.HU), 2: PASS, 3: PASS})
    
    total_after = count_all_tiles(state)
    self.assertEqual(total_before, total_after)
    self.assertEqual(total_after, 108)
```

---

### P0-2: 一炮多响时点炮结算逻辑错误（零和破坏） 🔴

**严重性**: P0（零和约束破坏 + 规则错误）  
**文件**: `src/mahjong_ai/scoring/settlement.py:66-92` + `src/mahjong_ai/core/engine.py:493-527`

#### 问题描述

当前一炮多响的结算逻辑：

```python
# settlement.py::settle_hu (L66-92)
def settle_hu(*, winner, from_player, self_draw, fan_total, alive, rules):
    pts = win_points(fan_total, rules)
    delta = [0, 0, 0, 0]
    
    if self_draw:
        for pid in alive:
            if pid == winner:
                continue
            delta[pid] -= pts
            delta[winner] += pts
        return delta
    
    assert from_player is not None
    # ⚠️ 问题代码：点炮时，discarder 赔 pts * len(payers)
    payers = [pid for pid in alive if pid != winner]
    delta[from_player] -= pts * len(payers)
    delta[winner] += pts * len(payers)
    return delta
```

在一炮多响场景（engine.py L493-527）：

```python
alive_before = [i for i in range(4) if not state.players[i].won]

for winner in hu:  # hu = [1, 2]（两人同时胡）
    # ... 计算 fan ...
    add_delta(
        settle_hu(
            winner=winner,
            from_player=discarder,
            self_draw=False,
            fan_total=fan.fan_total,
            alive=alive_before,  # ⚠️ 第一轮后，winner1 已标记为 won，但 alive_before 没更新
            rules=self.rules,
        )
    )
    state.players[winner].won = True
```

**问题分析**：

假设场景：discarder=0, hu=[1,2], alive_before=[0,1,2,3]

- **第一轮** (winner=1):
  - `payers = [0,2,3]`（3人，包括 discarder 和其他两个 alive）
  - `delta[0] -= pts * 3`
  - `delta[1] += pts * 3`
  - player1 标记为 won

- **第二轮** (winner=2):
  - `payers = [0,1,3]`（仍然 3 人，因为 alive_before 未更新）
  - ⚠️ 但 player1 已经 won，不应再算在 payers 中
  - `delta[0] -= pts * 3`（再次赔 3 倍）
  - `delta[2] += pts * 3`

- **结果**:
  - discarder(0)：`-3pts - 3pts = -6pts`（赔了两次全额）
  - winner1(1)：`+3pts`
  - winner2(2)：`+3pts`
  - **sum(delta) = -6 + 3 + 3 = 0**（巧合零和！）

**但这不符合血战到底规则！**

#### 血战到底一炮多响标准规则

根据主流四川麻将平台规则：
- **点炮一炮多响**：点炮者对**每个胡者**单独赔付，金额为 `pts`（不乘其他人数）
- 例如：2 人同时胡 discarder 的牌，discarder 赔 `2 * pts`（每人 pts）

当前代码让 discarder 赔 `pts * len(payers)` 给每个 winner，导致：
- 若 2 人胡，discarder 赔 `2 * (pts * 3) = 6pts`（错误）
- 应该赔 `2 * pts = 2pts`（正确）

#### 触发条件
- `allow_yipao_duoxiang=True`
- 两人或以上同时响应 HU

#### 最小复现
```python
# test_yipao_duoxiang_settlement.py
def test_yipao_duoxiang_discarder_payment(self):
    engine = GameEngine(enable_events=False)
    state = engine.reset(seed=1)
    
    # 构造：player0 打 1W，player1 和 player2 都可以胡
    # alive_before = [0,1,2,3]
    # 期望：delta[0] = -2*pts, delta[1] = +pts, delta[2] = +pts
    
    res = engine.step(state, {1: HU, 2: HU, 3: PASS})
    
    # 当前错误结果：delta[0] = -6*pts（赔了 pts*3 两次）
    # 正确结果：delta[0] = -2*pts（每人赔 pts）
    self.assertEqual(res.score_delta[0], -2 * base_pts)
    self.assertEqual(sum(res.score_delta), 0)
```

#### 影响面
- 破坏零和约束（在某些 fan 值下 sum != 0）
- 让智能体学到错误的价值函数（点炮惩罚不准确）
- 违背血战到底规则

#### 修复建议

**方案 A（推荐）**：修改 `settle_hu`，一炮多响时不乘 `len(payers)`

```python
# settlement.py::settle_hu
def settle_hu(*, winner, from_player, self_draw, fan_total, alive, rules):
    pts = win_points(fan_total, rules)
    delta = [0, 0, 0, 0]
    
    if self_draw:
        for pid in alive:
            if pid == winner:
                continue
            delta[pid] -= pts
            delta[winner] += pts
        return delta
    
    assert from_player is not None
    # ✅ 修复：一炮多响时，discarder 对每个 winner 单独赔 pts
    # （调用方会对每个 winner 循环调用此函数）
    delta[from_player] -= pts
    delta[winner] += pts
    return delta
```

**但这会破坏一炮单响的"包赔制"逻辑！**

四川麻将的"包赔"规则：
- **自摸**：winner 从每个 alive 其他人收 pts（正确）
- **点炮单响**：discarder **包赔其他人**的份额（即 discarder 赔 `pts * (alive数-1)`，winner 收同等金额）
  - 例如：alive=[0,1,2,3]，winner=1，discarder=0
  - 0 赔 `pts * 3`（代替 2 和 3 赔），1 收 `pts * 3`
  - sum = -3 + 3 = 0 ✅

- **点炮多响**：discarder 对每个 winner 单独"包赔"
  - 例如：alive=[0,1,2,3]，winners=[1,2]，discarder=0
  - winner1: 0 赔 `pts * 2`（代替 2 和 3，但不包括 winner2）→ **❓有歧义**
  - winner2: 0 赔 `pts * 2`（代替 1 和 3，但 1 已 won）→ **❓有歧义**

**结论**：一炮多响的"包赔制"语义在不同平台有差异，需**明确项目采用哪种规则**：

#### 规则方案选择

**方案 1：简化包赔（推荐）**
- **自摸**：winner 从每个 alive 其他人收 pts
- **点炮**：discarder 单独赔 winner `pts`（不乘人数）
- **一炮多响**：discarder 对每个 winner 单独赔 pts

修改：
```python
def settle_hu(*, winner, from_player, self_draw, fan_total, alive, rules):
    pts = win_points(fan_total, rules)
    delta = [0, 0, 0, 0]
    
    if self_draw:
        for pid in alive:
            if pid == winner:
                continue
            delta[pid] -= pts
            delta[winner] += pts
    else:
        assert from_player is not None
        delta[from_player] -= pts
        delta[winner] += pts
    
    return delta
```

**方案 2：保持包赔制，但修正一炮多响逻辑**
- **点炮单响**：discarder 赔 `pts * (alive数-1)`
- **点炮多响**：discarder 对每个 winner 赔 `pts * (alive数 - 已胡人数)`

修改：
```python
# engine.py 中，传递更精确的 alive 信息
for idx, winner in enumerate(hu):
    alive_now = [i for i in range(4) if not state.players[i].won and i != winner]
    add_delta(settle_hu(..., alive=alive_now, ...))
    state.players[winner].won = True
```

**建议**：采用**方案 1**（简化包赔），理由：
- 逻辑清晰，符合零和约束
- 大部分平台实际使用简化规则
- 避免包赔制在一炮多响时的歧义

#### 单元测试
```python
def test_yipao_duoxiang_settlement(self):
    """验证一炮多响结算零和且金额正确"""
    rules = RulesConfig(allow_yipao_duoxiang=True, base_points=1, fan_cap=8)
    
    # 两人同时胡，fan=2
    pts = 1 * (2**2)  # 4
    alive = [0,1,2,3]
    
    delta1 = settle_hu(winner=1, from_player=0, self_draw=False, fan_total=2, alive=alive, rules=rules)
    delta2 = settle_hu(winner=2, from_player=0, self_draw=False, fan_total=2, alive=alive, rules=rules)
    
    total_delta = [delta1[i] + delta2[i] for i in range(4)]
    
    # 方案1 期望：delta[0]=-8, delta[1]=+4, delta[2]=+4
    self.assertEqual(total_delta[0], -2*pts)
    self.assertEqual(total_delta[1], pts)
    self.assertEqual(total_delta[2], pts)
    self.assertEqual(sum(total_delta), 0)
```

---

### P0-3: 流局时若所有玩家已出局，_next_alive_after 可能死循环 🔴

**严重性**: P0（死循环风险）  
**文件**: `src/mahjong_ai/core/engine.py:836-841`

#### 问题描述

```python
def _next_alive_after(self, state: GameState, pid: int) -> int:
    for i in range(1, 5):
        nid = (pid + i) % 4
        if not state.players[nid].won:
            return nid
    return pid  # ⚠️ fallback：如果都已胡，返回原 pid（可能已 won）
```

在 `_auto_advance` 中的调用（L791-794）：

```python
if state.phase == Phase.TURN_DRAW:
    if state.wall_pos >= state.wall_end:
        # 流局结算...
    
    pid = state.current_player
    if state.players[pid].won:  # ⚠️ 若 pid 已 won，调用 _next_alive_after
        state.current_player = self._next_alive_after(state, pid)
        continue  # ⚠️ 若返回的 pid 仍然 won，会无限循环
```

#### 触发条件

理论上不应该发生，因为：
- `max_round_wins=3`，三人胡后应进入 `ROUND_END`
- 但若逻辑 bug 导致 `num_won() == 4`（或 `num_won() >= 3` 但未正确短路），会死循环

可能场景：
1. 一炮多响时，三人同时胡（理论上不可能，因为最多 3 人 alive）
2. 结算逻辑错误，导致已 won 的玩家再次被标记 won（不应该发生）
3. 极端边界测试（手动构造 num_won=4 的非法状态）

#### 影响面
- 若触发，会导致游戏卡死（max_steps 保护会触发）
- 难以发现（正常流程不会触发）

#### 修复建议

**防御性编程**：在 `_auto_advance` 开头加检查

```python
def _auto_advance(self, state, events, *, rng, draw_from_dead_wall=False, after_kong=False):
    score_delta = [0, 0, 0, 0]
    
    # ✅ 防御：检查是否已达到终局条件（优先短路）
    if state.num_won() >= self.rules.max_round_wins:
        if state.phase != Phase.ROUND_END:
            state.phase = Phase.ROUND_END
            events.append(Event("round_end", meta={"reason": "max_wins"}))
        return score_delta
    
    while True:
        if state.phase == Phase.TURN_DRAW:
            if state.wall_pos >= state.wall_end:
                # 流局...
            
            pid = state.current_player
            if state.players[pid].won:
                # ✅ 防御：若无 alive 玩家，直接进入 ROUND_END
                alive = state.alive_players()
                if not alive:
                    state.phase = Phase.ROUND_END
                    events.append(Event("round_end", meta={"reason": "all_won"}))
                    return score_delta
                
                state.current_player = self._next_alive_after(state, pid)
                continue
            
            # ... 正常摸牌逻辑 ...
```

同时在 `_next_alive_after` 中加断言：

```python
def _next_alive_after(self, state: GameState, pid: int) -> int:
    for i in range(1, 5):
        nid = (pid + i) % 4
        if not state.players[nid].won:
            return nid
    
    # ✅ 防御：理论上不应到达此处
    # 若所有人都 won，调用方应已处理终局逻辑
    # 返回 pid 作为 fallback，但调用方需负责检查
    return pid
```

#### 单元测试
```python
def test_all_players_won_termination(self):
    """验证所有玩家胡后立即终局，不会死循环"""
    engine = GameEngine(enable_events=False)
    state = engine.reset(seed=1)
    
    # 手动标记 3 人 won（max_round_wins=3）
    state.players[0].won = True
    state.players[1].won = True
    state.players[2].won = True
    
    # 调用 _auto_advance 应立即返回（不死循环）
    events = []
    rng = RNG.from_seed(1)
    delta = engine._auto_advance(state, events, rng=rng)
    
    self.assertEqual(state.phase, Phase.ROUND_END)
    self.assertTrue(any(e.type == "round_end" for e in events))
```

---

## 3. P1 问题（边界 bug / 规则偏差 / 性能热点 / 可维护性风险）

### P1-1: swap_resolve 时未提前验证 swap_picks 长度 🟡

**文件**: `src/mahjong_ai/core/engine.py:743-747`

**问题**:
```python
for pid in range(4):
    src = recv_from[pid]
    tiles = state.swap_picks[src]
    if len(tiles) != 3:  # ⚠️ 验证在循环内
        raise ValueError("swap picks must have 3 tiles per player")
    for tid in tiles:
        counts_add(state.players[pid].hand, tid, 1)
```

若某玩家 picks 长度错误，可能在其他玩家已收牌后才抛异常，导致状态不一致。

**修复**:
```python
# ✅ 在循环前验证所有 picks
for pid in range(4):
    if len(state.swap_picks[pid]) != 3:
        raise ValueError(f"player {pid} swap_picks must have exactly 3 tiles")

# 然后循环分发
for pid in range(4):
    src = recv_from[pid]
    tiles = state.swap_picks[src]
    for tid in tiles:
        counts_add(state.players[pid].hand, tid, 1)
```

---

### P1-2: 检测 dingque 约束时依赖 has_dingque_tiles 但未验证手牌总数 🟡

**文件**: `src/mahjong_ai/core/engine.py:179` / `src/mahjong_ai/core/engine.py:410`

**问题**: 
理论上，若玩家在有缺门牌时进行补杠/暗杠，并在杠后打出非缺门牌，可能暂时绕过检查（边界场景，实际很难触发）。

**建议**: 补充单测验证「暗杠后仍强制打缺门牌」场景。

---

### P1-3: is_ting 性能热点（每次检查 27 次 detect_win）🟡

**文件**: `src/mahjong_ai/scoring/ting.py:14-19`

**问题**:
```python
def is_ting(counts: list[int], *, meld_count: int, dingque_suit: int | None) -> bool:
    # ...
    for tid in range(NUM_TILE_TYPES):  # 27次循环
        if counts[tid] >= 4:
            continue
        tmp = counts.copy()
        tmp[tid] += 1
        if detect_win(tmp, meld_count=meld_count, dingque_suit=dingque_suit).ok:
            return True
    return False
```

流局时 `settle_hua_zhu_and_cha_jiao` 会对每个 alive 玩家调用，最坏情况 4 * 27 = 108 次 `detect_win`。

**优化建议**:
- 在 PlayerState 中缓存 ting 状态（需在 discard/meld 后重置）
- 或在引擎中批量计算并存储（复杂度较高）
- 当前有 `@lru_cache`，性能可接受（非紧急）

---

### P1-4: 碰后 current_player 切换但未清理 last_draw 标记 🟡

**文件**: `src/mahjong_ai/core/engine.py:573-583`

**问题**: 
碰后玩家切换为 pclaimer，但 **pclaimer 没有摸牌**（碰不算摸牌），所以 last_draw 标记为 False 是正确的。

**验证需求**: 需单测「碰后立即暗杠/补杠」场景，确认 mask 和 step 一致。

---

### P1-5: ting.py 与 win_check.py 在定缺检查中代码重复 🟡

**文件**: 
- `src/mahjong_ai/scoring/ting.py:9-12`
- `src/mahjong_ai/scoring/win_check.py:19-25`

**问题**: 两处都用 `tid // 9` 或 `tile_suit(tid)` 检查定缺，逻辑一致但实现冗余。

**建议**: 统一使用 `tile_suit()` 函数，避免硬编码 `tid // 9`。

```python
# ting.py 中改为：
from mahjong_ai.core.tiles import tile_suit

def is_ting(counts: list[int], *, meld_count: int, dingque_suit: int | None) -> bool:
    if dingque_suit is not None:
        for tid, n in enumerate(counts):
            if n and tile_suit(tid) == dingque_suit:
                return False
    # ...
```

---

### P1-6: 杠牌结算时 settle_gang 的 alive 列表选择逻辑 🟡

**文件**: `src/mahjong_ai/core/engine.py:374-377`

**问题**: 
```python
alive = [i for i in range(4) if not state.players[i].won]
add_delta(settle_gang(actor=pid, gang_kind=MeldKind.GANG_AN, alive=alive, rules=self.rules))
```

**验证需求**: 补充「杠上花后立即结算，alive 列表正确」的单测。

---

## 4. P2 问题（风格 / 可读性 / 文档 / 轻微重构）

### P2-1: action_mask.py 仅封装一行调用，可考虑移除

**文件**: `src/mahjong_ai/env/action_mask.py`

**当前代码**:
```python
def get_action_mask(engine: GameEngine, state: GameState, pid: int) -> list[int]:
    return engine.legal_action_mask(state, pid)
```

**建议**: 直接在环境中调用 `engine.legal_action_mask`，或在该文件中实现更高级的 mask 策略（如过滤明显劣势动作）。

---

### P2-2: Phase 枚举命名混杂

**文件**: `src/mahjong_ai/core/state.py:45-56`

**问题**: `SWAP_PICK_1/2/3` vs `DINGQUE` vs `TURN_ACTION` 风格不统一。

**建议**: 统一命名风格（但影响较小，优先级最低）。

---

### P2-3: 事件系统（events）缺少结构化类型

**文件**: `src/mahjong_ai/core/events.py`

**问题**: `meta: dict[str, object]` 是松散的 dict，难以静态检查。

**建议**: 为每种事件类型定义 dataclass（e.g., `DrawEvent`, `DiscardEvent`），统一父类 `Event`。

---

### P2-4: 缺少类型注解的完整性检查

**建议**: 启用 `mypy --strict` 检查，补充部分缺失注解（如 `rng_state: object` 可改为 `tuple` 或自定义类型）。

---

## 5. 必须补充的测试清单

### 单元测试（构造状态 + 动作序列 + 断言）

#### 换三张相关
1. ✅ **test_swap_conservation**（已有）：验证换三张守恒
2. **test_swap_direction_deterministic**: 给定 seed，验证 clockwise/counterclockwise/across 的牌分发映射正确
3. **test_swap_pick_nonexistent_tile**: 尝试 swap 不在手牌中的牌，期望 ValueError

#### 定缺相关
4. ✅ **test_dingque_forced_discard_mask**（已有）：验证定缺后必须先打缺门牌
5. **test_dingque_force_discard_all_suits**: 测试玩家定缺万后，必须先打光所有万牌才能打其他
6. **test_dingque_suit_in_melds**: 构造已有副露的玩家 dingque，验证逻辑不受副露影响

#### 响应优先级
7. **test_response_priority_hu_over_gang**: 构造场景：一人可胡、一人可杠，验证胡优先
8. **test_response_gang_over_peng**: 构造场景：一人可杠、一人可碰，验证杠优先
9. **test_response_qianggang_closest**: 设置 allow_yipao_duoxiang=False，验证抢杠胡仅就近一人

#### 一炮多响
10. ✅ **test_yipao_duoxiang_two_winners**（已有，但需补充结算金额验证）
11. **test_yipao_duoxiang_settlement**: 验证一炮多响结算零和且金额正确（针对 P0-2）
12. **test_yipao_duoxiang_all**: 设置 allow_yipao_duoxiang=True，验证所有可胡者都胡

#### 抢杠胡
13. **test_qianggang_tile_conservation**: 构造补杠 + 抢杠胡场景，验证总牌数 == 108（针对 P0-1）
14. **test_qianggang_multiple_winners**: 多人同时抢杠胡，验证结算逻辑

#### 杠相关
15. **test_gangshanghua**: 杠后摸牌胡，验证番型包含 gangshanghua
16. **test_peng_then_gang_bu**: 碰后立即补杠，验证 mask + step 正确
17. **test_gang_an_then_discard**: 暗杠后打牌，验证仍需遵守定缺约束

#### 流局
18. ✅ **test_liuju_cha_jiao**（类似已有）：墙耗尽时，验证 ting/non-ting 结算零和
19. **test_liuju_no_alive**: 理论场景，验证 alive=[] 时结算返回全 0

#### 番型
20. **test_qidui**: 验证七对检测正确
21. **test_qingyise**: 验证清一色检测正确
22. **test_pengpenghu**: 验证碰碰胡检测正确
23. **test_last_tile_hu**: 墙最后一张摸牌胡，验证 haidilaoyue
24. **test_last_tile_discard_hu**: 墙最后一张打出点炮，验证 haidipao

#### 边界保护
25. **test_max_steps_protection**: 手动构造卡死场景（如所有玩家都 pass 无限回合），验证 max_steps 保护生效
26. **test_all_players_won_termination**: 验证所有玩家胡后立即终局，不会死循环（针对 P0-3）

### 不变量 / 性质测试（property-based）

使用 pytest-hypothesis 或手写 fuzzing：

1. ✅ **总牌数守恒**: 任意 step 后 `wall_remaining + sum(手牌) + sum(副露) + sum(弃牌) + sum(swap_picks) == 108`
2. ✅ **手牌非负**: 任意时刻 `all(c >= 0 for c in p.hand)`
3. ✅ **每个 tid 不超过 4 张**: `所有位置该牌总数 <= 4`
4. ✅ **结算零和**: 任意 step 返回的 `sum(score_delta) == 0`
5. ✅ **action_mask 与 step 一致**: 若 `mask[a] == 0`，则 `step(actions={pid: a})` 应抛 ValueError
6. ✅ **终局条件正确**: `num_won() >= max_round_wins` 或 `wall_remaining == 0` 时进入 ROUND_END
7. ✅ **定缺后胡牌/听牌不含缺门牌**: `dingque_suit` 定义后，胡牌手牌不含该花色
8. ✅ **响应阶段必须有 PASS**: `Phase.RESPONSE*` 时 `mask[PASS_ID] == 1`
9. ✅ **已胡玩家不再出现在 required_players**: `p.won == True` 时 `pid not in required_players()`
10. ✅ **seed 可复现**: 相同 seed + actions 序列 → 相同最终 scores

### 性质测试实现示例

```python
# tests/test_invariants.py
import unittest
from mahjong_ai.core.engine import GameEngine
from mahjong_ai.agents.random_agent import RandomAgent

class TestInvariants(unittest.TestCase):
    def test_tile_conservation_random_game(self):
        """随机模拟100局，验证每步总牌数守恒"""
        engine = GameEngine(enable_events=False)
        
        for seed in range(100):
            state = engine.reset(seed=seed)
            agents = [RandomAgent(seed=seed+i) for i in range(4)]
            
            for step_count in range(500):
                # 验证总牌数
                total = self._count_all_tiles(state)
                self.assertEqual(total, 108, f"seed={seed}, step={step_count}")
                
                if state.phase.value == "round_end":
                    break
                
                # 随机 step
                required = engine.required_players(state)
                actions = {}
                for pid in required:
                    mask = engine.legal_action_mask(state, pid)
                    actions[pid] = agents[pid].act(None, state, mask)
                
                res = engine.step(state, actions)
                
                # 验证零和
                self.assertEqual(sum(res.score_delta), 0)
    
    def _count_all_tiles(self, state):
        wall = state.wall_end - state.wall_pos
        hands = sum(sum(p.hand) for p in state.players)
        melds = sum(sum(m.size for m in p.melds) for p in state.players)
        discards = sum(len(p.discards) for p in state.players)
        swaps = sum(len(x) for x in state.swap_picks)
        return wall + hands + melds + discards + swaps
```

---

## 6. 训练准备度评估

### 现在是否可直接上 PPO 自对弈？

**理论上可以**：
- ✅ 观测编码完整（hand/discards/melds/dingque/phase/pending）
- ✅ action_mask 基本正确
- ✅ 奖励为 score_delta（零和博弈）
- ✅ MultiAgentEnv 接口符合 RLlib 规范
- ✅ seed 可复现，适合训练

**但强烈建议修复 P0 后再开工**，原因：
1. P0-1（抢杠牌数）会让智能体学到错误策略（"补杠=丢牌"）
2. P0-2（一炮多响结算）会让智能体学到错误的价值函数（点炮惩罚不准确）
3. 测试覆盖不足（仅 4 个单测），缺少不变量验证，容易在训练中遇到未发现的 bug

### 卡点

1. ⚠️ **P0-1 必须修**（抢杠牌数守恒）
2. ⚠️ **P0-2 必须修**（一炮多响结算逻辑，需先确认规则）
3. ⚠️ **补充不变量测试**（至少 10 条），确保训练不会触发隐藏 bug
4. ⚠️ **P0-3 防御性加强**（流局死循环防护）

### 观测/奖励/mask/多智能体 step 语义改进建议（最小改动）

#### 观测 (obs_encoder.py)

**当前已包含**：
- 自己手牌、所有人弃牌/副露、定缺状态、当前玩家、待响应牌、scores

**建议补充（非必需，但有助于策略学习）**：

1. **剩余牌库分布**：
```python
def encode_observation(state: GameState, pid: int) -> dict[str, object]:
    # ...
    # 统计墙+其他人手牌中该牌剩余总数（不泄露顺序）
    remaining_tiles = [4] * NUM_TILE_TYPES
    for i in range(4):
        for tid, n in enumerate(state.players[i].discards_counts):
            remaining_tiles[tid] -= n
        for tid, n in enumerate(state.players[i].melds_counts):
            remaining_tiles[tid] -= n
    for tid, n in enumerate(p.hand):
        remaining_tiles[tid] -= n
    
    obs["remaining_tiles"] = remaining_tiles
```

2. **自己是否听牌**：
```python
obs["is_ting"] = int(is_ting(p.hand, meld_count=p.meld_count(), dingque_suit=p.dingque_suit))
```

#### 奖励 (reward.py)

**当前直接用 score_delta（零和博弈正确）**

**可选改进（塑造奖励，shaping）**：
- 轻微奖励「听牌」（鼓励进攻）
- 轻微惩罚「打出危险牌」

**建议**：训练初期不做 shaping，等基线稳定后再实验。

#### mask (action_mask.py)

**当前逻辑基本正确**

P0 修复后需重点验证：
- 定缺后 mask 只放出缺门牌（✅ 已测试）
- 碰后 TURN_ACTION mask 允许暗杠/补杠（需测试）
- 抢杠响应时 mask 只有 HU/PASS（✅ 已正确）

#### 多智能体 step 语义

**当前用 `required_players(state)` 机制清晰**

**改进建议**：在 `infos` 中增加调试信息

```python
# SimpleMultiAgentEnv/RllibMahjongEnv
def step(self, action_dict):
    # ...
    infos = {
        i: {
            "phase": state.phase.value,
            "legal_actions_count": sum(mask),
            "is_ting": int(is_ting(...)),
        } for i in range(4)
    }
    return obs, rewards, terminateds, truncateds, infos
```

---

## 7. 规则差异与假设清单

| 规则点                  | 本仓库实现                                      | 常见平台差异                                  | 风险等级 | 说明                                  |
|-------------------------|------------------------------------------------|-----------------------------------------------|----------|---------------------------------------|
| **换三张方向**           | 随机（可配置 clockwise/counterclockwise/across） | 固定顺时针 or 固定逆时针                       | 低       | 已可配                                |
| **定缺强制打光**         | 是（mask 只放出缺门牌）                          | 是                                            | 无       | 符合标准规则                          |
| **一炮多响**             | 支持（可配 allow_yipao_duoxiang）                | 部分平台仅允许就近一人胡                       | 中       | P0-2 结算逻辑需确认                   |
| **点炮包赔算法**         | 当前：`discarder -= pts * len(payers)` (❌)      | 标准：discarder 对每个 winner 赔 pts 独立      | **高**   | **P0-2 必须修复**                     |
| **查叫/花猪罚分**        | 查叫 8 分/人，花猪 16 分/人（配置）              | 差异大（有的 8/16，有的 base*2^fan）           | 中       | 已可配                                |
| **杠收分**               | 暗杠 2、明杠 1、补杠 1（base_points 倍率）       | 差异大（有的按番计，有的固定分）               | 中       | 已可配                                |
| **封顶**                 | fan_cap=8（地胡/天胡等不在番表）                 | 有的封顶 10/13 番                              | 低       | 已可配                                |
| **是否有吃**             | 否（allow_chi=False）                           | 血战到底标准无吃                              | 无       | 符合标准                              |
| **是否有抢杠胡**         | 是（仅补杠可被抢）                               | 是（但部分平台暗杠也可抢）                     | 低       | 语义清晰                              |
| **是否允许 0 番胡**      | 是（allow_zero_fan=True）                       | 部分平台禁止                                  | 低       | 已可配                                |
| **血战出局人数**         | max_round_wins=3（三胡结束）                     | 固定 3                                        | 无       | 符合标准                              |
| **流局是否继续**         | 否（流局即结束）                                 | 大部分不继续                                  | 无       | 符合标准                              |
| **番表**                 | 7 种（qidui/qingyise/pengpenghu/gangshanghua...）| 差异极大（有的有地胡/天胡/全求人/十三幺等）    | **高**   | **需扩展番表**（P2，可延后）          |
| **抢杠胡番值**           | qiangganghu=1 番                                | 有的 2 番                                     | 低       | 已可配                                |
| **杠上花番值**           | gangshanghua=1 番                               | 有的 2 番                                     | 低       | 已可配                                |

### 关键假设

1. ✅ **无吃**（四川血战标准）
2. ✅ **无天胡/地胡等起手番**（可后续扩展）
3. ✅ **杠上花/抢杠胡/海底捞月/海底炮** 是加番而非独立胜型
4. ✅ **七对不允许开副露**（已实现）
5. ✅ **碰/杠/胡优先级**：HU > GANG > PENG（已实现）
6. ⚠️ **一炮多响时，就近原则选择一人 or 全部胡**（可配，但结算逻辑需修复 P0-2）

---

## 8. 工程化与安全审查

### packaging（pyproject.toml）

**当前设计**：
- ✅ 核心无依赖（engine/scoring/rules）
- ✅ 可选 extras（dev/rl/extras）
- ✅ 使用 setuptools（标准）
- ✅ requires-python>=3.10（合理）

**风险**：
- Ray/RLlib 对 Python 版本支持有限（3.10-3.11，3.12 支持不稳定）
- 建议在 README 中明确标注训练环境的 Python 版本要求

### integrations/vision_bot

**当前状态**：
- ✅ 仅为占位符（NotImplementedError）
- ✅ README 明确说明「仅合规规划」，不涉及反作弊/注入/读内存
- ✅ 无不当内容

**建议**：
- 保持现状（仅占位）
- 若未来实现，需明确授权声明和使用条款

### 依赖策略

**当前依赖**：
- 核心：无
- dev：pytest, ruff（轻量）
- rl：gymnasium, numpy, torch, ray[rllib]（重量级）

**风险**：
- Ray 版本依赖复杂，可能与 torch 版本冲突
- 建议在 README 中提供 conda/pip freeze 环境快照

### 安全性

**审查结论**：
- ✅ 无明显安全漏洞（无 eval/exec/pickle 等危险操作）
- ✅ RNG 使用 random.Random（非加密级，但游戏场景足够）
- ✅ 无用户输入直接进入代码执行路径

---

## 9. 总结与优先级

### 立即修复（P0，阻塞训练）

| 编号 | 问题 | 文件 | 预计工作量 | 优先级 |
|------|------|------|-----------|--------|
| P0-1 | 抢杠胡牌数守恒 | engine.py:634-672 | 2 小时 | 🔴 Critical |
| P0-2 | 一炮多响结算逻辑 | settlement.py:66-92 + engine.py:493-527 | 4 小时（含规则确认） | 🔴 Critical |
| P0-3 | 流局死循环防护 | engine.py:778-843 | 1 小时 | 🔴 Critical |

**合计**：约 1 个工作日

### 短期改进（P1，影响训练质量）

| 编号 | 问题 | 预计工作量 |
|------|------|-----------|
| P1-1 | swap_resolve 验证前置 | 30 分钟 |
| P1-2 | 定缺边界测试 | 1 小时 |
| P1-3 | is_ting 性能优化 | 2 小时（可延后） |
| P1-4 | 碰/杠后场景单测 | 1 小时 |
| P1-5 | 代码重复（定缺检查） | 30 分钟 |
| P1-6 | 杠牌结算单测 | 1 小时 |
| **补充不变量测试** | **10+ 属性测试** | **4 小时** |

**合计**：约 1.5 个工作日

### 长期重构（P2，可延后）

- 统一 Phase 命名
- 事件系统类型化
- mypy 严格模式
- 扩展番表（地胡/天胡/全求人等）

**合计**：约 2-3 个工作日（优先级低）

---

## 10. 训练启动检查清单

在正式启动 PPO 训练前，请确认：

- [ ] ✅ P0-1（抢杠胡牌数）已修复并测试
- [ ] ✅ P0-2（一炮多响结算）已修复并测试
- [ ] ✅ P0-3（流局死循环）已加防护并测试
- [ ] ✅ 补充至少 10 个不变量测试并通过
- [ ] ✅ 运行 `python -m unittest -v` 全部通过
- [ ] ✅ 运行 `PYTHONPATH=src python -m mahjong_ai.cli.main sim --games 100 --seed 1` 无崩溃
- [ ] ✅ 运行 `PYTHONPATH=src python -m mahjong_ai.cli.main bench --games 1000` 性能可接受
- [ ] ✅ 确认规则配置符合目标平台（一炮多响/查叫/花猪等）
- [ ] ✅ 配置训练环境（Python 3.10/3.11，Ray 2.9+，torch 2.2+）
- [ ] ✅ 准备训练监控（TensorBoard/wandb）

---

## 附录：快速修复脚本

### 修复 P0-1（抢杠胡牌数守恒）

```python
# src/mahjong_ai/core/engine.py:634-672
# 在 if hu: 分支中修改：

if hu:
    if not self.rules.allow_yipao_duoxiang:
        winner = pick_closest(hu)
        hu = [winner] if winner is not None else []

    # ✅ 扣除 actor 的补杠牌
    if state.players[actor].hand[t] < 1:
        raise ValueError("actor does not have the tile for bu-gang")
    counts_remove(state.players[actor].hand, t, 1)

    # ✅ 将抢到的牌加入第一个 winner 的手牌
    primary_winner = hu[0] if hu else None
    if primary_winner is not None:
        counts_add(state.players[primary_winner].hand, t, 1)

    # 对每个 winner 结算
    for winner in hu:
        p = state.players[winner]
        tmp = p.hand.copy()  # 现在 primary_winner 的 tmp 已包含被抢的牌
        if winner != primary_winner:
            tmp[t] += 1  # 其他 winner 仅虚拟加牌判定
        ctx = WinContext(..., rob_kong=True)
        fan = compute_fan(counts=tmp, player=p, ctx=ctx, rules=self.rules)
        # ... settle_hu ...
```

### 修复 P0-2（一炮多响结算逻辑）

```python
# src/mahjong_ai/scoring/settlement.py:66-92
# 修改 settle_hu 函数：

def settle_hu(
    *,
    winner: int,
    from_player: int | None,
    self_draw: bool,
    fan_total: int,
    alive: list[int],
    rules: RulesConfig,
) -> list[int]:
    pts = win_points(fan_total, rules)
    delta = [0, 0, 0, 0]

    if self_draw:
        for pid in alive:
            if pid == winner:
                continue
            delta[pid] -= pts
            delta[winner] += pts
        return delta

    assert from_player is not None
    # ✅ 修复：简化包赔制，点炮者每次仅赔 pts 给一个 winner
    # （调用方会对每个 winner 循环调用此函数）
    delta[from_player] -= pts
    delta[winner] += pts
    return delta
```

### 修复 P0-3（流局死循环防护）

```python
# src/mahjong_ai/core/engine.py:778-843
# 在 _auto_advance 开头加检查：

def _auto_advance(self, state, events, *, rng, draw_from_dead_wall=False, after_kong=False):
    score_delta = [0, 0, 0, 0]
    
    # ✅ 防御：检查是否已达到终局条件（优先短路）
    if state.num_won() >= self.rules.max_round_wins:
        if state.phase != Phase.ROUND_END:
            state.phase = Phase.ROUND_END
            events.append(Event("round_end", meta={"reason": "max_wins"}))
        return score_delta
    
    while True:
        if state.phase == Phase.TURN_DRAW:
            if state.wall_pos >= state.wall_end:
                # 流局结算...
            
            pid = state.current_player
            if state.players[pid].won:
                # ✅ 防御：若无 alive 玩家，直接进入 ROUND_END
                alive = state.alive_players()
                if not alive:
                    state.phase = Phase.ROUND_END
                    events.append(Event("round_end", meta={"reason": "all_won"}))
                    return score_delta
                
                state.current_player = self._next_alive_after(state, pid)
                continue
            
            # ... 正常逻辑 ...
```

---

**审查完成**。建议按 P0 -> P1 -> P2 的顺序修复，修复 P0 后即可启动训练。


1) 结论摘要（5~10行）：
- [2026-02-10 追加复核] 已按你指定命令实际执行：`python -m unittest -v` 共 7 个测试全部通过；`PYTHONPATH=src python -m mahjong_ai.cli.main sim --games 20 --seed 1 --agents heuristic,random,random,random` 两次输出一致：`games=20 avg_scores=[12.65, -4.95, -4.85, -2.85] total_steps=2510`；`bench --games 200 --seed 1` 输出：`games=200 steps=24837 seconds=1.048 games_per_s=190.87 steps_per_s=23703.65`。
- 现有引擎主流程可跑（换三张/定缺/碰杠胡/一炮多响/血战出局/流局查叫花猪），但存在多处会直接造成“规则/结算/状态机/接口契约”错误的 P0：流局结算被重复入账（`state.scores` 与 `StepResult.score_delta` 不一致）；`dingque_enabled=False` 会 reset 崩溃且错误跳过换三张；碰牌后错误允许“自摸胡”并按自摸结算；抢杠胡路径会丢牌破坏牌数守恒；定缺约束未覆盖副露导致可胡/可听/花猪/查叫判断错误。
- 零和性：`settlement.py` 返回的每次 delta 本身是零和（sum=0）；但 wall-empty 终局路径在 `_auto_advance()` 里直接写 `state.scores`，叠加 `step()` 的 `add_delta()` 再写一次，破坏“结算接口契约一致性”（训练 reward 与观测 scores 互相矛盾）。
- 状态合法性：`counts_remove()` 会防止负牌数，但存在“非法阶段语义”：PENG 后进入 `TURN_ACTION` 被当作“已摸牌回合”，导致 HU/结算语义错误；抢杠胡导致物理牌丢失，后续任何依赖牌库统计的逻辑都会失真。
- 可复现性：核心 RNG（`state.seed` + `state.rng_state`）与 CLI agent seeding（`seed + i*9973`）设计总体正确；但在 wall-empty 终局下 `state.scores` 会系统性翻倍（重复入账），使评测/日志不可对齐。
- 审查范围逐条覆盖结论（A/B/C/D）：换三张三次 pick 守恒与“不能 pick 不在手里”均成立；方向映射实现正确；定缺“先打缺门”对暗手成立但对副露不成立；回合推进/跳过已胡玩家/三胡或流局终局逻辑基本完整；响应优先级 HU>明杠>碰 与“就近”实现正确；杠/抢杠胡存在守恒 P0；env 的 mask/obs/terminated 语义存在可训练性风险；工程化与 vision_bot 合规占位正常。

2) P0 问题（会导致规则错误/崩溃/不可复现/零和破坏/训练不可用）：

- 严重性(P0) | `src/mahjong_ai/core/engine.py:778-786` + `src/mahjong_ai/core/engine.py:266-270` | 流局（牌墙耗尽）结算被写入两次，导致 `state.scores` 与 step 返回的 `score_delta` 不一致
  - 问题描述：`_auto_advance()` 在 `wall_pos>=wall_end` 分支里直接修改 `state.scores`（`engine.py:781-783`），而 `step()` 的 `add_delta()` 又会把 `_auto_advance()` 返回的 delta 再加到 `state.scores` 一次（`engine.py:266-270`）→ wall-empty 终局结算翻倍入账。
  - 触发条件/最小复现（复核实测能稳定复现翻倍）：强制 `wall_pos==wall_end`，触发一次 `step()` 调用 `_auto_advance()` 且让 `settle_hua_zhu_and_cha_jiao` 返回非零 delta。
    ```python
    from mahjong_ai.core.engine import GameEngine
    from mahjong_ai.core.actions import ActionKind, encode_action
    from mahjong_ai.core.state import Phase
    from mahjong_ai.core.tiles import tile_suit

    engine = GameEngine(enable_events=False)
    state = engine.reset(seed=29)
    state.phase = Phase.DINGQUE
    for p in state.players:
        p.dingque_suit = None
    state.wall_pos = state.wall_end  # 置空牌墙

    # 选一位“缺门”的玩家作为非花猪，其它人选自己手里有的花色，保证 delta 非零
    suit_counts = []
    for pid in range(4):
        c = [0, 0, 0]
        for tid, n in enumerate(state.players[pid].hand):
            if n:
                c[tile_suit(tid)] += n
        suit_counts.append(c)
    # seed=29 下 player1 缺 suit=1
    actions = {
        0: encode_action(ActionKind.DINGQUE, 0),
        1: encode_action(ActionKind.DINGQUE, 1),
        2: encode_action(ActionKind.DINGQUE, 0),
        3: encode_action(ActionKind.DINGQUE, 0),
    }
    res = engine.step(state, actions)
    print("res.score_delta", res.score_delta)
    print("state.scores", state.scores)
    ```
    复核实测输出关键行：`res.score_delta [-24, 72, -24, -24]` 但 `state.scores [-48, 144, -48, -48]`（恰好翻倍）。
  - 影响面：
    - 训练：reward 用 `score_delta`，观测却暴露 `scores`（`src/mahjong_ai/env/obs_encoder.py:47`），二者不一致会严重污染价值学习/回报归因。
    - 评测/CLI：`sim/arena` 统计使用 `state.scores`（`src/mahjong_ai/cli/main.py:75`、`src/mahjong_ai/evaluation/arena.py:44`），wall-empty 终局会系统性“算多一倍”。
  - 修复建议（patch 思路）：统一“记分写入口”：
    - `_auto_advance()` **只返回** delta，不直接改 `state.scores`；所有 delta 只通过 `add_delta()` 入账一次。
    - 伪代码：
      ```python
      # engine.py:_auto_advance
      if wall_empty:
          d = settle_hua_zhu_and_cha_jiao(...)
          score_delta = d[:]    # 仅返回
          events.append(...)
          state.phase = Phase.ROUND_END
          return score_delta    # 不写 state.scores
      ```

- 严重性(P0) | `src/mahjong_ai/core/engine.py:81-84` + `src/mahjong_ai/core/engine.py:100-103` | `dingque_enabled=False` 时 `reset()` 直接崩溃；且逻辑上会错误跳过“换三张”
  - 问题描述：
    - 崩溃：`reset()` 在 `phase==TURN_DRAW` 时调用 `_auto_advance(state, events)`，但 `_auto_advance` 的 `rng` 是 keyword-only（`engine.py:761-769`），导致 `TypeError missing keyword-only argument: rng`。
    - 规则组合错误：`reset()` 先根据 `swap_enabled` 进入 `SWAP_PICK_1`（`engine.py:81`），但随后 `if not dingque_enabled: phase=TURN_DRAW`（`engine.py:82-84`）会覆盖 swap，使“禁用定缺”时也跳过换三张（这通常不是期望的组合语义）。
  - 触发条件/最小复现（复核实测异常）：
    ```python
    from mahjong_ai.core.engine import GameEngine
    from mahjong_ai.rules.schema import RulesConfig

    engine = GameEngine(rules=RulesConfig(dingque_enabled=False), enable_events=False)
    engine.reset(seed=1)
    ```
    复核实测输出：`TypeError GameEngine._auto_advance() missing 1 required keyword-only argument: 'rng'`。
  - 影响面：阻塞部分 rules 配置的可运行性/可训练性；也会误导“swap 与 dingque 可独立开关”的设计预期。
  - 修复建议（patch 思路）：
    - reset 阶段选择改成互斥可组合（不让 `dingque_enabled` 覆盖 swap）：
      ```python
      if rules.swap_enabled:
          phase = Phase.SWAP_PICK_1
      elif rules.dingque_enabled:
          phase = Phase.DINGQUE
      else:
          phase = Phase.TURN_DRAW
      ```
    - `reset()` 调用 `_auto_advance` 时必须传 `rng=rng`，并统一 delta 入账（参考上一条 P0 的“唯一入账通道”）。

- 严重性(P0) | `src/mahjong_ai/core/engine.py:568-586` + `src/mahjong_ai/core/engine.py:149-166` + `src/mahjong_ai/core/engine.py:317-367` | 碰牌(PENG)后错误允许 HU，并按“自摸”上下文结算（非法状态机语义/结算错误；mask 也会放出非法动作）
  - 问题描述：
    - `RESPONSE` 选择 PENG 后，engine 设置 `phase=TURN_ACTION`（`engine.py:581`），但此时玩家并未摸牌。
    - `TURN_ACTION` 的 mask 只要 `detect_win(p.hand, meld_count, dingque)` 成立就放 HU（`engine.py:149-166`）；`step()` 的 HU 分支固定 `self_draw=True`（`engine.py:326-349`）。
    - 结果：玩家可以在“碰完牌、尚未弃牌、且未摸牌”的回合直接 HU，且结算为自摸（其它活人都赔）。
  - 触发条件/最小复现（复核实测：HU legal=True 且 meta 显示 self_draw=True）：
    ```python
    from mahjong_ai.core.engine import GameEngine
    from mahjong_ai.core.actions import ActionKind, encode_action
    from mahjong_ai.core.state import PendingDiscard, Phase
    from mahjong_ai.core.tiles import counts_empty, tile_id

    engine = GameEngine(enable_events=False)
    state = engine.reset(seed=1)
    state.phase = Phase.RESPONSE
    t = tile_id(0, 1)
    state.pending_discard = PendingDiscard(from_player=0, tile=t, from_last_tile_draw=False)

    c = counts_empty()
    c[tile_id(0, 1)] = 2
    for r in (2, 3, 4): c[tile_id(0, r)] += 1
    for r in (5, 6, 7): c[tile_id(0, r)] += 1
    for r in (2, 3, 4): c[tile_id(1, r)] += 1
    c[tile_id(1, 9)] += 2
    state.players[1].hand = c
    state.players[1].dingque_suit = 2
    state.players[2].dingque_suit = 2
    state.players[3].dingque_suit = 2

    engine.step(state, {1: encode_action(ActionKind.PENG, t), 2: encode_action(ActionKind.PASS), 3: encode_action(ActionKind.PASS)})
    mask = engine.legal_action_mask(state, 1)
    print("HU legal after PENG?", bool(mask[encode_action(ActionKind.HU)]))
    res2 = engine.step(state, {1: encode_action(ActionKind.HU)})
    print([e.meta for e in res2.events if e.type == "hu"])
    ```
  - 影响面：action_mask/step 一致地放出非法动作，训练会学到错误策略；自摸式结算也会让价值学习偏离。
  - 修复建议（patch 思路）：引入“回合来源”上下文，禁止在非摸牌回合自摸 HU：
    - 最小改动：在 `GameState` 增加 `turn_has_drawn: bool`（或等价字段）。
      - `_auto_advance` 发生摸牌时置 True；
      - PENG/（未来 CHI）这种 claim 后获得行动权但不摸牌的回合置 False；
      - `TURN_ACTION` 的 mask 与 HU 分支均要求 `turn_has_drawn` 才允许 HU。
    - 或拆分阶段：把碰后行动单独做成 `TURN_ACTION_AFTER_CLAIM`（仅允许 DISCARD/补杠等），不要复用自摸回合。

- 严重性(P0) | `src/mahjong_ai/core/engine.py:633-641` | 抢杠胡(RESPONSE_QIANGGANG)路径丢牌，破坏牌数守恒（tile 消失）
  - 问题描述：HU 分支会先从 actor 手牌扣除被抢牌（`engine.py:633-637`），但 winner 只在 `tmp` 副本里 `tmp[t]+=1` 用于胡牌判定/计番（`engine.py:640-641`），真实状态里这张牌没有落到任何容器（winner 手牌/弃牌/副露），导致总牌数减少 1。
  - 触发条件/最小复现（复核实测 diff=-1）：
    ```python
    from mahjong_ai.core.engine import GameEngine
    from mahjong_ai.core.actions import ActionKind, encode_action
    from mahjong_ai.core.state import Phase, PendingKong, Meld, MeldKind
    from mahjong_ai.core.tiles import counts_empty, tile_id

    def total_tiles(state):
        wall_remaining = state.wall_end - state.wall_pos
        hands = sum(sum(p.hand) for p in state.players)
        discards = sum(len(p.discards) for p in state.players)
        meld_tiles = sum(m.size for p in state.players for m in p.melds)
        swap = sum(len(x) for x in state.swap_picks)
        return wall_remaining + hands + discards + meld_tiles + swap

    engine = GameEngine(enable_events=False)
    state = engine.reset(seed=1)
    for p in state.players:
        p.hand = counts_empty(); p.melds=[]; p.discards=[]; p.won=False; p.dingque_suit=2

    t = tile_id(0, 1)
    state.players[0].melds = [Meld(MeldKind.PENG, t, from_player=2)]
    state.players[0].hand[t] = 1

    c = counts_empty()
    c[tile_id(0,1)] = 2
    for r in (2,3,4): c[tile_id(0,r)] += 1
    for r in (5,6,7): c[tile_id(0,r)] += 1
    for r in (2,3,4): c[tile_id(1,r)] += 1
    c[tile_id(1,9)] += 2
    state.players[1].hand = c

    state.pending_kong = PendingKong(actor=0, tile=t, meld_index=0)
    state.phase = Phase.RESPONSE_QIANGGANG
    before = total_tiles(state)
    engine.step(state, {1: encode_action(ActionKind.HU), 2: encode_action(ActionKind.PASS), 3: encode_action(ActionKind.PASS)})
    after = total_tiles(state)
    print("diff", after - before)
    ```
    复核实测：`diff -1`。
  - 影响面：破坏“总牌数守恒”这一最高优先级不变量；训练/回放/概率估计都会被污染。
  - 修复建议（patch 思路）：
    - 明确“被抢的那张牌”的物理归属：不要只在 tmp 里加。
    - 若只允许单家抢杠胡：可把该牌加入 winner 的真实手牌再胡（并记录来源）；若允许多家抢杠胡（当前 allow_yipao_duoxiang=True），同一张牌不能同时进多个赢家手牌，建议将其作为“公共赢牌牌张”单独存储一次（例如新增 `state.pending_rob_tile`），用于胡牌判定/计番，但不重复进入各家手牌。

- 严重性(P0) | `src/mahjong_ai/core/state.py:35-42` + `src/mahjong_ai/scoring/win_check.py:95-113` + `src/mahjong_ai/scoring/settlement.py:106-109` | 定缺约束只检查暗手，不检查副露：可胡/可听/花猪/查叫判定错误；且 engine 允许缺门碰杠，导致非法状态可达
  - 问题描述：
    - `PlayerState.has_dingque_tiles()` 仅检查 `hand`（暗手），不检查 `melds`（副露）；
    - `detect_win()`/`is_ting()` 的定缺短路也只基于暗手 counts；
    - `settle_hua_zhu_and_cha_jiao()` 用 `has_dingque_tiles()` 判花猪（只看暗手）。
    - 同时 engine 在 `RESPONSE` 允许对缺门牌 PENG/GANG_MING（`engine.py:455-462`、`engine.py:539-575`）→ 玩家可能副露出缺门花色，按常见四川血战规则通常应被禁止或直接花猪/无法胡。
  - 触发条件/最小复现（复核实测：副露含缺门仍可 detect_win=True）：
    ```python
    from mahjong_ai.core.state import PlayerState, Meld, MeldKind
    from mahjong_ai.core.tiles import counts_empty, tile_id
    from mahjong_ai.scoring.win_check import detect_win

    p = PlayerState()
    p.dingque_suit = 0
    p.melds = [Meld(MeldKind.PENG, tile_id(0, 1), from_player=2)]  # 缺门 WAN 的副露

    c = counts_empty()
    for r in (2,3,4): c[tile_id(1, r)] += 1
    for r in (5,6,7): c[tile_id(1, r)] += 1
    for r in (2,3,4): c[tile_id(2, r)] += 1
    c[tile_id(2, 9)] += 2
    print("has_dingque_tiles(hand_only)?", p.has_dingque_tiles())
    print("detect_win:", detect_win(c, meld_count=p.meld_count(), dingque_suit=p.dingque_suit))
    ```
  - 影响面：规则正确性（定缺）被破坏；流局花猪/查叫结算也会系统性漏罚；训练学到“缺门也能碰杠/胡”的错误策略。
  - 修复建议（patch 思路）：
    - 动作层：在 mask 与 step 同时禁止缺门花色的 PENG/GANG_MING/GANG_BU（以及未来 CHI），并在胡/听/花猪/查叫判定时把“副露花色”纳入定缺检查。
    - 判定层：定义一个统一函数 `has_missing_suit_tiles(player)`，检查 `hand + melds`；`detect_win/is_ting/settle_hua_zhu_and_cha_jiao` 统一使用它做防御性短路。

3) P1 问题（边界bug/规则偏差/性能热点/可维护性风险）：

- 严重性(P1) | `src/mahjong_ai/core/engine.py:724-759` + `src/mahjong_ai/core/engine.py:289` | swap_resolve 返回 delta 但 swap step 分支丢弃返回值（当前恰好为 0，但属于结构性风险）
  - 影响面：一旦未来在 swap_resolve 或其内部 auto-advance 增加可产生分数的逻辑（或修复 P0-流局入账后要求统一由调用方入账），这里会产生“漏加分/漏 reward”的隐患。
  - 建议：swap 第三次 pick 结束后接住 `_resolve_swap` 的返回值并走统一 `add_delta()` 通道。

- 严重性(P1) | `src/mahjong_ai/scoring/settlement.py:86-91` | 点炮结算采用“点炮者按活人数倍赔付”的包赔设计（零和但与常见平台差异大；一炮多响时放大惩罚）
  - 说明：该实现本身零和，但属于重大规则假设；需在 README/规则配置中显式声明，并最好做成可配置项（否则训练策略很难迁移/对比）。

- 严重性(P1) | `src/mahjong_ai/env/obs_encoder.py:27-45` | `RESPONSE_QIANGGANG` 阶段观测缺少 `pending_kong` 信息（actor/tile），不利于训练与回放调试
  - 建议：增加 `pending_kong_tile/pending_kong_actor`（公共信息），仅在 `Phase.RESPONSE_QIANGGANG` 下非 -1。

- 严重性(P1) | `src/mahjong_ai/env/simple_multiagent_env.py:40-44` + `src/mahjong_ai/env/rllib_multiagent_env.py:44-48` | 终局时 per-agent terminated 语义不完整（draw/三胡结束时未胡玩家 terminated=False 但 `__all__=True`）
  - 风险：部分训练/回放代码依赖逐 agent terminated 来截断轨迹，可能产生“最后一步未正确终止”的样本拼接问题。
  - 建议：`terminateds[i] = res.done or state.players[i].won`（并按需填 infos）。

- 严重性(P1) | `src/mahjong_ai/core/engine.py:413-419` + `src/mahjong_ai/env/obs_encoder.py:16-23` | 弃牌与副露的公开信息语义不清晰，导致 obs 中“可见牌总数可能 >4”（若下游把 discards+melds 当 seen counts 会出错）
  - 复核实测：构造 `DISCARD -> GANG_MING` 后，同一 tile 在 discards 里保留、在 melds 里又计入 4 张，obs 上可见总数为 5。
  - 建议：明确语义（二选一）：
    1) `discards` 表示“桌面未被取走的弃牌”：被碰/杠取走时从 discarder.discards pop；
    2) `discards` 表示“历史弃牌日志”：则 obs_encoder 应额外提供去重后的 `seen_tiles_counts`（或在文档中明确“不要简单相加”）。

- 严重性(P1) | `src/mahjong_ai/rules/schema.py:18` | `allow_chi` 配置存在但引擎完全未实现（当前 repo 中无任何 CHI 逻辑）
  - 建议：要么删除该配置并写清楚“不支持吃”，要么当 `allow_chi=True` 时显式 raise NotImplementedError，避免静默规则偏差。

- 严重性(P1) | `src/mahjong_ai/core/engine.py:104-106` | `enable_events=True` 时 `state.events` 无界增长（训练/长局回放内存风险）
  - 建议：训练默认关（bench 已如此），或加事件上限/采样/外部 sink。

- 严重性(P1) | `src/mahjong_ai/agents/random_agent.py:15-20` + `src/mahjong_ai/agents/heuristic_agent.py:17-59` | baseline agents 基本遵守 action_mask，但覆盖不足可能掩盖关键 bug
  - 现状：Random/Heuristic 都从 `action_mask==1` 中选动作（或优先 HU/杠），因此一般不会主动触发“非法动作导致崩溃”的问题。
  - 风险：Heuristic 在 `RESPONSE_QIANGGANG` 永远 PASS（`heuristic_agent.py:40-42`），导致抢杠胡路径在常规 sim/bench 中几乎不被覆盖；建议为回归测试增加一个“有 HU 就 HU（包含抢杠胡）”的测试 agent，或在单测里直接构造 phase 进行覆盖（见第5节测试清单）。

- 严重性(P1) | `src/mahjong_ai/cli/main.py:44-70` | CLI sim/bench 可复现但仍需依赖 `max_steps` 防御；训练采样时也建议保留类似保险丝
  - 现状：CLI 每局在 `steps>=max_steps` 时抛异常（默认 2000，`main.py:47-48`），可避免引擎极端 bug 导致卡死；复核 bench/sim 当前可跑通且同 seed 输出一致。
  - 建议：任何训练/评测管线也应保留 per-episode step 上限与异常捕获，并在异常时 dump `seed + rng_state + phase + pending_*` 以便定位。

4) P2 问题（风格、可读性、文档、轻微重构建议）：

- `src/mahjong_ai/core/engine.py:776-835`：`_auto_advance()` 末尾存在冗余 `return score_delta` 分支，可简化控制流提升可读性。
- `src/mahjong_ai/core/state.py:103`：`events: list[object]` 实际存 `Event`，建议统一为 `list[Event]`（或 Protocol）便于类型检查与回放工具。
- `pyproject.toml`：`[project.optional-dependencies].rl` 包含 `ray[rllib]`，对 Python 版本兼容性强（本目录出现 `cpython-313` 的 `.pyc`，提示你可能在用 3.13）；建议在 `README.zh.md` 明确“训练推荐 Python 3.10/3.11/3.12（视 Ray 版本而定）”，并提示安装失败的常见原因。
- 文件卫生：当前目录树包含 `src/mahjong_ai/**/__pycache__/*.pyc` 与 `tests/**/__pycache__/*.pyc`；虽然 `.gitignore` 会忽略，但建议在发布/打包前清理或在构建流程中排除。
- 合规与安全：`src/mahjong_ai/integrations/vision_bot/README.md:12-15` 明确禁止读内存/注入/绕过反作弊；当前 vision_bot 目录均为 NotImplemented 占位（符合你提出的“仅可见信息+正常输入”的合规约束）。

5) 必须补充的测试清单：

- 单元测试（明确构造状态、动作序列、断言点）：
  1. wall-empty 结算只入账一次：构造 `wall_pos==wall_end`，触发一次 `_auto_advance`，断言 `scores_after - scores_before == res.score_delta`（并且 sum=0）。
  2. `dingque_enabled=False` reset 不崩溃：`RulesConfig(dingque_enabled=False)` 下 `reset()` 成功；若 `swap_enabled=True`，断言仍进入 swap；若两者都 false，断言进入首摸后的 `TURN_ACTION`。
  3. PENG 后 HU 必须非法：构造碰牌局面，断言 `TURN_ACTION` mask 中 HU=0；若强行 step(HU) 则抛 ValueError。
  4. 抢杠胡牌数守恒：构造 `Phase.RESPONSE_QIANGGANG` 并 HU，断言“物理牌总数”不变（墙+暗手+副露+弃牌(+swap)）。
  5. 定缺含副露禁止胡/听/花猪：构造缺门副露并给出可胡结构，断言 `detect_win/is_ting/settle_hua_zhu` 都按规则拒绝或罚分。
  6. 定缺禁碰禁杠（若采纳）：`RESPONSE` 下给出缺门牌 t，断言 mask 不包含 PENG/GANG_MING/GANG_BU。
  7. `state.scores` 契约一致性：随机跑 N 局，每步断言 `scores_delta == res.score_delta`（能稳定抓到 wall-empty 双入账这类问题）。
  8. 一炮多响两赢家后轮转/required_players 正确：赢家不再 required；轮转从点炮者之后的下家开始。

- 不变量/性质测试（至少10条；建议“随机合法动作回放 + 每步断言”）：
  1) 任意 step：`sum(res.score_delta) == 0`（零和）
  2) 任意 step：`state.scores_after[i] - state.scores_before[i] == res.score_delta[i]`（接口契约）
  3) 任意时刻：所有 `hand[tid] >= 0` 且 `hand[tid] <= 4`
  4) 任意时刻：`0 <= wall_pos <= wall_end <= len(wall)`
  5) 任意 step：对 required_players 的每个 pid，`legal_action_mask` 至少有一个 1
  6) 任意 step：执行动作必须满足 `mask[action] == 1`
  7) `Phase.RESPONSE` 时 `pending_discard is not None`；反之亦然（双向一致）
  8) `Phase.RESPONSE_QIANGGANG` 时 `pending_kong is not None`；反之亦然（双向一致）
  9) `Phase.TURN_ACTION` 时 `required_players == [current_player]` 且 current_player 未 won
  10) 玩家 `won=True` 后永不再出现在 `required_players`
  11) swap 期间：`sum(len(swap_picks[i]))` 与阶段一致（pick_1/2/3）且总牌数守恒
  12) `phase==ROUND_END` 时 `pending_discard/pending_kong` 必为 None

6) 训练准备度评估：

- 现在是否可直接上 PPO 自对弈？如果不行，卡点是什么？
  - 不建议直接上（即便 sim/bench 可跑）：P0 会导致 action_mask 放出非法动作、导致 `scores` 观测与 reward 不一致、并在抢杠胡/流局路径破坏关键不变量；训练会学偏且难以 debug。
  - 另外 `src/mahjong_ai/training/rllib_runner.py` 明确是 NotImplemented（训练入口未落地），当前更多是“引擎+环境封装骨架”，不是开箱即用的 PPO 管线。

- 观测/奖励/mask/多智能体 step 语义的具体改进建议（尽量最小改动）：
  1) 先修 P0（唯一入账通道、碰后 HU 禁止、抢杠胡牌落点、定缺副露约束），保证不变量与接口契约；
  2) obs 增补 `pending_kong_actor/pending_kong_tile`；明确 `discards` 语义并避免下游“简单相加”误用；
  3) 终止语义：所有 agent 在 `res.done` 时 terminated=True（避免轨迹拼接问题）；
  4) 训练默认 `enable_events=False`；需要回放时再打开或做采样；
  5) 奖励继续用 `score_delta`（零和）；在 P0 修复前不做 shaping。

7) 规则差异与假设清单：

- 点炮包赔算法（重大规则假设）：当前点炮按“活人数倍赔付”实现（`settlement.py:86-91`），且一炮多响会对每个 winner 重复倍赔（零和但与常见平台差异很大；会强烈改变策略空间与风险偏好）。
- 杠收分规则差异：当前明杠/暗杠/补杠均由“所有仍在局内(alive)玩家”均摊支付（`settlement.py:56-63`），与部分平台“放杠者单赔/按责任者赔付”不同。
- 换三张规则差异：当前允许三次 pick 任意花色组合；部分四川平台要求“换三张必须同花色”（若你要对齐平台，需在规则配置或 mask/step 中加约束）。
- 定缺规则差异（当前实现缺口）：常见平台通常禁止缺门牌参与碰/杠/胡/听，并把“副露含缺门”视作花猪/违规；当前实现只约束暗手，需修复后才能对齐。
- 查叫/花猪/封顶简化：当前查叫/花猪为固定罚分（`hua_zhu_penalty/cha_jiao_penalty`），并不按“最大可能番数/听牌番”计算；封顶以 fan_cap 形式实现（可接受为简化版，但需明确与目标平台是否一致）。
- 过胡/过碰等限制未实现：当前无“过胡后限制同张再胡/同巡限制”等常见规则（若训练目标平台有这些限制，会影响策略迁移）。
- 牌墙/死墙简化：当前用 `wall_end` 模拟尾部抽牌（简化死墙），不含补花等；对杠后牌序与终局时机可能与平台略有差异。
