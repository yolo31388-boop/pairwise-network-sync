"""网络状态同步。

修复点：
1. send_position/receive_position 走快照缓冲 + 渲染插值，客户端不再瞬移；
   同步频率随移动速度自适应（默认基线 10Hz，快速移动时自动提频）。
2. client_prediction 只把输入放进待确认队列，server_correction 做服务器
   纠正与回滚：被拒输入（如未生效的技能）回滚并返回，供客户端退还 CD。
3. compress_state 用 half-float(float16) 压缩坐标并附带 CRC32 校验和，
   decompress_state 校验失败会丢弃数据包。
4. lag_compensation 同时计入攻击方与被攻击方延迟（双向/RTT 口径）。
5. reconnect_sync 基于版本号做增量同步，首连才下发全量基线。
"""
from __future__ import annotations

import math
import struct
import time
import zlib
from collections import deque
from dataclasses import dataclass, field, asdict

# 渲染插值延迟：用约两个网络帧之前的位置做插值，掩盖网络抖动
INTERP_DELAY = 0.1
# 位置快照插值缓冲上限
SNAPSHOT_HISTORY = 16
# 全量基线之后，单次增量允许携带的字段
_DELTA_FIELDS = ("x", "y", "velocity_x", "velocity_y", "seq")


def _to_half(value: float) -> float:
    """把 float 量化到 IEEE 754 half-precision，再转回 float。"""
    return struct.unpack("<e", struct.pack("<e", float(value)))[0]


@dataclass
class PlayerState:
    pid: str
    x: float = 0.0
    y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    seq: int = 0
    version: int = 0


class CorruptStateError(ValueError):
    """状态包校验和不匹配时抛出。"""


class NetworkSync:
    def __init__(self, base_send_rate: float = 10.0, interp_delay: float = INTERP_DELAY):
        self.states: dict[str, PlayerState] = {}
        # bug2 修复：待服务器确认的预测输入；prediction_log 只记录已确认输入
        self.pending_inputs: dict[str, list[dict]] = {}
        self.prediction_log: list[dict] = []
        self.rolled_back_actions: list[dict] = []
        self._input_seq: dict[str, int] = {}

        # 快照插值缓冲：pid -> deque[{"seq","x","y","vx","vy","ts"}]
        self._snapshots: dict[str, deque] = {}
        self._last_sent_packet: dict[str, dict] = {}
        self._last_send_ts: dict[str, float] = {}
        self.base_send_rate = base_send_rate
        self.interp_delay = interp_delay

        # 增量重连：每个客户端已确认到的版本号 + 权威状态历史
        self._acked_version: dict[str, int] = {}
        self._version_history: dict[str, dict[int, dict]] = {}

    # ------------------------------------------------------------------
    # 1. 位置同步：快照 + 插值 + 自适应发送频率
    # ------------------------------------------------------------------
    def send_position(self, pid: str, x: float, y: float,
                      timestamp: float | None = None) -> dict:
        """权威端（服务器/本地玩家）产生位置包。

        位置进入快照缓冲，渲染状态由 update() 插值推进，不会直接瞬移。
        """
        now = time.monotonic() if timestamp is None else timestamp
        state = self.states.get(pid)
        if state is None:
            state = PlayerState(pid, x=float(x), y=float(y))
            self.states[pid] = state
            self._snapshots[pid] = deque([{
                "seq": 0, "x": float(x), "y": float(y),
                "vx": 0.0, "vy": 0.0, "ts": now,
            }], maxlen=SNAPSHOT_HISTORY)

        buf = self._snapshots.setdefault(
            pid, deque(maxlen=SNAPSHOT_HISTORY))
        prev = buf[-1] if buf else None
        dt = max(now - prev["ts"], 1e-6) if prev else 0.0
        vx = (float(x) - prev["x"]) / dt if prev and dt > 0 else state.velocity_x
        vy = (float(y) - prev["y"]) / dt if prev and dt > 0 else state.velocity_y
        seq = (prev["seq"] + 1) if prev else 0
        snapshot = {"seq": seq, "x": float(x), "y": float(y),
                    "vx": vx, "vy": vy, "ts": now}
        buf.append(snapshot)

        state.seq = seq
        state.velocity_x, state.velocity_y = vx, vy
        state.version += 1
        history = self._version_history.setdefault(pid, {})
        history[state.version] = {
            "x": float(x), "y": float(y),
            "velocity_x": vx, "velocity_y": vy, "seq": seq,
        }
        for old_version in sorted(history)[:-SNAPSHOT_HISTORY]:
            del history[old_version]

        packet = {"pid": pid, "seq": seq, "x": float(x), "y": float(y),
                  "velocity_x": vx, "velocity_y": vy, "ts": now}
        self._last_sent_packet[pid] = packet
        self._last_send_ts[pid] = now
        return packet

    def receive_position(self, packet: dict) -> None:
        """客户端收到位置包：只入快照缓冲，由 update() 插值消费。"""
        pid = packet["pid"]
        buf = self._snapshots.setdefault(pid, deque(maxlen=SNAPSHOT_HISTORY))
        if buf and packet["seq"] <= buf[-1]["seq"]:
            return  # 过期/乱序包
        buf.append({
            "seq": packet["seq"],
            "x": float(packet["x"]),
            "y": float(packet["y"]),
            "vx": float(packet.get("velocity_x", 0.0)),
            "vy": float(packet.get("velocity_y", 0.0)),
            "ts": float(packet.get("ts", time.monotonic())),
        })
        if pid not in self.states:
            self.states[pid] = PlayerState(pid)

    def update(self, dt: float, now: float | None = None) -> None:
        """渲染帧推进：沿快照缓冲把所有角色的渲染位置插值到 render_time。"""
        if now is None:
            now = time.monotonic()
        render_time = now - self.interp_delay
        for pid, buf in self._snapshots.items():
            if not buf:
                continue
            state = self.states.setdefault(pid, PlayerState(pid))
            # 找到 render_time 落在哪两个快照之间
            after = None
            for snap in buf:
                if snap["ts"] >= render_time:
                    after = snap
                    break
            if after is None:
                newest = buf[-1]
                # 缓冲不够深：按速度做短暂外推，而不是瞬移
                gap = max(render_time - newest["ts"], 0.0)
                state.x = newest["x"] + newest["vx"] * gap
                state.y = newest["y"] + newest["vy"] * gap
                continue
            idx = buf.index(after)
            if idx == 0:
                before = after
            else:
                before = buf[idx - 1]
            span = after["ts"] - before["ts"]
            alpha = 0.0 if span <= 0 else (render_time - before["ts"]) / span
            alpha = min(max(alpha, 0.0), 1.0)
            state.x = before["x"] + (after["x"] - before["x"]) * alpha
            state.y = before["y"] + (after["y"] - before["y"]) * alpha

    def tick(self, dt: float = 1.0 / 30.0) -> None:
        self.update(dt)

    def send_interval(self, speed: float) -> float:
        """自适应同步间隔：速度越快间隔越短（频率越高），限制在合理范围。"""
        base = 1.0 / self.base_send_rate
        if speed <= 0:
            return base
        # 保证每个包之间的位移大致恒定：快速移动时自动提频
        interval = min(base, max(1.0 / 60.0, 2.0 / max(speed, 1e-6)))
        return interval

    def should_send(self, pid: str, speed: float, now: float | None = None) -> bool:
        if pid not in self._last_send_ts:
            return True
        now = time.monotonic() if now is None else now
        return (now - self._last_send_ts[pid]) >= self.send_interval(speed)

    # ------------------------------------------------------------------
    # 2. 客户端预测 + 服务器纠正回滚
    # ------------------------------------------------------------------
    def client_prediction(self, pid: str, action: str,
                          snapshot: dict | None = None) -> dict:
        """本地先预测执行，输入进入待确认队列（尚未计入已确认日志）。"""
        seq = self._input_seq.get(pid, 0) + 1
        self._input_seq[pid] = seq
        entry = {"pid": pid, "seq": seq, "action": action,
                 "snapshot": snapshot}
        self.pending_inputs.setdefault(pid, []).append(entry)
        return entry

    def server_correction(self, pid: str, last_ack_seq: int | None = None,
                          accepted: list[str] | None = None,
                          corrected: PlayerState | dict | None = None) -> list[dict]:
        """处理服务器纠正。

        - last_ack_seq 之前的输入视为服务器确认；
        - 之后的输入全部回滚（服务器说技能没放 -> 返回这些动作供客户端退 CD）；
        - accepted 中服务器认可的输入在纠正后的状态上重放；
        - 客户端状态回退到服务器权威状态。
        """
        pending = self.pending_inputs.get(pid, [])
        confirmed, rejected = [], []
        for entry in pending:
            if last_ack_seq is not None and entry["seq"] <= last_ack_seq:
                confirmed.append(entry)
            elif accepted is not None and entry["action"] in accepted:
                confirmed.append(entry)
            else:
                rejected.append(entry)

        if corrected is not None:
            self._apply_authoritative_state(pid, corrected)

        self.prediction_log.extend(confirmed)
        self.rolled_back_actions.extend(rejected)
        self.pending_inputs[pid] = []
        return rejected

    # 常见命名别名，方便不同调用方
    def correct_prediction(self, *args, **kwargs):
        return self.server_correction(*args, **kwargs)

    def rollback(self, *args, **kwargs):
        return self.server_correction(*args, **kwargs)

    def _apply_authoritative_state(self, pid, corrected) -> None:
        state = self.states.setdefault(pid, PlayerState(pid))
        if isinstance(corrected, PlayerState):
            state.x, state.y = corrected.x, corrected.y
            state.velocity_x = corrected.velocity_x
            state.velocity_y = corrected.velocity_y
        else:
            for attr in ("x", "y", "velocity_x", "velocity_y"):
                if attr in corrected:
                    setattr(state, attr, float(corrected[attr]))

    # ------------------------------------------------------------------
    # 3. 状态压缩：half-float + CRC32 校验和
    # ------------------------------------------------------------------
    def compress_state(self, state: PlayerState) -> dict:
        payload = {
            "pid": state.pid,
            "x": _to_half(state.x),
            "y": _to_half(state.y),
            "vx": _to_half(state.velocity_x),
            "vy": _to_half(state.velocity_y),
            "seq": int(state.seq),
        }
        payload["checksum"] = self._checksum(payload)
        return payload

    def decompress_state(self, packet: dict) -> PlayerState:
        """解压并校验；校验和不匹配直接拒绝（CorruptStateError）。"""
        packet = dict(packet)
        received_sum = packet.pop("checksum", None)
        if received_sum is None or received_sum != self._checksum(packet):
            raise CorruptStateError("state packet failed checksum verification")
        return PlayerState(
            pid=str(packet["pid"]),
            x=float(packet["x"]),
            y=float(packet["y"]),
            velocity_x=float(packet.get("vx", 0.0)),
            velocity_y=float(packet.get("vy", 0.0)),
            seq=int(packet.get("seq", 0)),
        )

    @staticmethod
    def _checksum(payload: dict) -> int:
        body = repr(sorted((k, v) for k, v in payload.items()
                           if k != "checksum")).encode("utf-8")
        return zlib.crc32(body) & 0xFFFFFFFF

    # ------------------------------------------------------------------
    # 4. 延迟补偿：双向（攻击方 + 被攻击方）
    # ------------------------------------------------------------------
    def lag_compensation(self, attacker_latency: int,
                         target_latency: int) -> float:
        """回溯时间（秒）同时计入两端延迟，等价于半 RTT 的双向口径。"""
        return (max(0, int(attacker_latency)) +
                max(0, int(target_latency))) / 1000.0

    # ------------------------------------------------------------------
    # 5. 断线重连：版本号增量同步
    # ------------------------------------------------------------------
    def reconnect_sync(self, pid: str,
                       since_version: int | None = None) -> dict:
        state = self.states.get(pid)
        if state is None:
            state = PlayerState(pid)
            self.states[pid] = state

        baseline = self._authoritative_baseline(pid, state)
        last_acked = self._acked_version.get(pid)
        if since_version is None:
            since_version = last_acked

        if since_version is None:
            # 首连：下发全量基线
            payload = dict(baseline)
            payload["pid"] = pid
            payload["version"] = state.version
            payload["incremental"] = False
            self._acked_version[pid] = state.version
            return {pid: payload, "version": state.version,
                    "incremental": False}

        if since_version >= state.version:
            # 没有更新：空增量，不重发全量
            payload = {"pid": pid, "version": state.version,
                       "incremental": True}
            return {pid: payload, "version": state.version,
                    "incremental": True}

        # 增量：只带相对上次版本真正变化的字段
        delta = {"pid": pid, "version": state.version,
                 "incremental": True}
        old = self._version_history.get(pid, {}).get(since_version, {})
        changed = {f: baseline[f] for f in _DELTA_FIELDS
                   if old.get(f) != baseline[f]}
        delta.update(changed)
        self._acked_version[pid] = state.version
        return {pid: delta, "version": state.version, "incremental": True}

    def acknowledge_sync(self, pid: str, version: int) -> None:
        self._acked_version[pid] = version

    def _authoritative_baseline(self, pid: str, state: PlayerState) -> dict:
        history = self._version_history.get(pid)
        if history:
            return dict(history[max(history)])
        return {"x": state.x, "y": state.y,
                "velocity_x": state.velocity_x,
                "velocity_y": state.velocity_y, "seq": state.seq}
