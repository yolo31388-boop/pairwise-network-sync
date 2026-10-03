"""
网络同步与状态压缩系统 - 核心模块

修复内容：
- state_synchronization(): 增量同步，只同步变化的玩家；按优先级分级；按频率控制同步节奏
- delta_compression(): 字段级增量压缩，只发送变化的字段，无变化不发送
- interpolation(): 快照插值，渲染时刻考虑插值延迟与网络延迟，丢包时按速度外推（有上限）
- packet_loss_recovery(): 基于序列号的丢包检测、NACK 重传、按序列号的字段级状态合并
- bandwidth_optimization(): QoS 优先级调度 + 基于令牌桶的流量控制（考虑接收方处理能力）
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time


PRIORITY_LOW = 0
PRIORITY_MEDIUM = 1
PRIORITY_HIGH = 2

QOS_LOW = "low"
QOS_NORMAL = "normal"
QOS_CRITICAL = "critical"

_PRIORITY_TO_QOS = {
    PRIORITY_LOW: QOS_LOW,
    PRIORITY_MEDIUM: QOS_NORMAL,
    PRIORITY_HIGH: QOS_CRITICAL,
}

# 各优先级的同步频率（Hz）：重要的状态发得勤，不重要的状态发得少
_DEFAULT_PRIORITY_RATES = {
    PRIORITY_HIGH: 20.0,
    PRIORITY_MEDIUM: 10.0,
    PRIORITY_LOW: 2.0,
}

_PACKET_HEADER_SIZE = 32
_SNAPSHOT_BUFFER_SIZE = 32
_EPS = 1e-9


@dataclass
class Config:
    sync_rate: float = 20.0
    interpolation_delay: float = 0.1
    extrapolation_limit: float = 0.25
    bandwidth_limit: float = 64 * 1024
    receiver_capacity: float = 64 * 1024


@dataclass
class Packet:
    player_id: str
    seq: int
    timestamp: float
    base_seq: int
    changes: Dict[str, Any]
    priority: int = PRIORITY_MEDIUM
    qos: str = QOS_NORMAL
    reliable: bool = True
    size: int = 0


def _lerp(a: Any, b: Any, t: float) -> Any:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + (b - a) * t
    return b if t >= 1.0 else a


class NetworkSync:
    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(config or {})
        self.sync_rate = float(self.config.get("sync_rate", 20.0))
        configured_rates = self.config.get("priority_rates", {})
        self._priority_rates = dict(_DEFAULT_PRIORITY_RATES)
        self._priority_rates.update(configured_rates)
        self.interpolation_delay = float(self.config.get("interpolation_delay", 0.1))
        self.extrapolation_limit = float(self.config.get("extrapolation_limit", 0.25))
        self.bandwidth_limit = float(self.config.get("bandwidth_limit", 64 * 1024))
        self.receiver_capacity = float(self.config.get("receiver_capacity", self.bandwidth_limit))

        # 发送方状态
        self._state: Dict[str, Dict] = {}
        self._last_sent: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        self._last_sync_time: Dict[int, float] = {}
        self._send_seq: Dict[str, int] = {}
        self._history: Dict[Tuple[str, int], Packet] = {}
        self._pending_acks: Dict[Tuple[str, int], Packet] = {}

        # 接收方状态
        self._recv_seq: Dict[str, int] = {}
        self._missing: Dict[str, set] = {}
        self._remote_state: Dict[str, Dict[str, Any]] = {}
        self._remote_field_seq: Dict[str, Dict[str, int]] = {}
        self._snapshots: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        self._latency: Dict[str, float] = {}

        # 流量控制令牌桶
        self._tokens = self._flow_rate()
        self._token_time: Optional[float] = None

    # ------------------------------------------------------------------
    # 本地状态维护
    # ------------------------------------------------------------------
    def update_player_state(self, player_id: str, fields: Dict[str, Any],
                            priority: int = PRIORITY_MEDIUM):
        entry = self._state.setdefault(player_id, {"fields": {}, "priority": priority, "dirty": False})
        entry["priority"] = priority
        for key, value in fields.items():
            if key not in entry["fields"] or entry["fields"][key] != value:
                entry["fields"][key] = value
                entry["dirty"] = True

    def update(self, dt: float):
        if dt and dt > 0:
            rate = self._flow_rate()
            self._tokens = min(rate, self._tokens + rate * dt)

    def reset(self):
        self._state = {}
        self._last_sent = {}
        self._last_sync_time = {}
        self._send_seq = {}
        self._history = {}
        self._pending_acks = {}
        self._recv_seq = {}
        self._missing = {}
        self._remote_state = {}
        self._remote_field_seq = {}
        self._snapshots = {}
        self._latency = {}
        self._tokens = self._flow_rate()
        self._token_time = None

    # ------------------------------------------------------------------
    # 1. 状态同步：增量 + 优先级 + 频率控制
    # ------------------------------------------------------------------
    def state_synchronization(self, now: Optional[float] = None) -> List[Packet]:
        """按优先级各自的频率，把发生变化的状态以增量包的形式发出去。"""
        now = time.time() if now is None else now
        packets: List[Packet] = []
        for priority, rate in sorted(self._priority_rates.items(), reverse=True):
            interval = 1.0 / rate if rate > 0 else float("inf")
            last = self._last_sync_time.get(priority, float("-inf"))
            if now - last < interval - _EPS:
                continue
            due = [pid for pid, s in self._state.items()
                   if s["priority"] == priority and s["dirty"]]
            if not due:
                continue
            self._last_sync_time[priority] = now
            for pid in sorted(due):
                delta = self.delta_compression(pid)
                if not delta["changes"]:
                    self._state[pid]["dirty"] = False
                    continue
                seq = self._send_seq.get(pid, 0) + 1
                self._send_seq[pid] = seq
                packet = Packet(
                    player_id=pid,
                    seq=seq,
                    timestamp=now,
                    base_seq=delta["base_seq"],
                    changes=delta["changes"],
                    priority=priority,
                    qos=_PRIORITY_TO_QOS.get(priority, QOS_NORMAL),
                )
                packet.size = _PACKET_HEADER_SIZE + len(repr(packet.changes).encode("utf-8"))
                packets.append(packet)
                self._history[(pid, seq)] = packet
                self._pending_acks[(pid, seq)] = packet
                self._last_sent[pid] = (seq, dict(self._state[pid]["fields"]))
                self._state[pid]["dirty"] = False
        return packets

    # ------------------------------------------------------------------
    # 2. 增量压缩：字段级差异，只发送变化量
    # ------------------------------------------------------------------
    def delta_compression(self, player_id: str,
                          current: Optional[Dict[str, Any]] = None,
                          baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """对指定玩家的状态做字段级增量压缩，返回相对基线的变化字段。"""
        if current is None:
            current = self._state.get(player_id, {}).get("fields", {})
        if baseline is None:
            base_seq, baseline = self._last_sent.get(player_id, (0, {}))
        else:
            base_seq = self._last_sent.get(player_id, (0, {}))[0]
        changes = {k: v for k, v in current.items()
                   if k not in baseline or baseline[k] != v}
        return {"player_id": player_id, "base_seq": base_seq, "changes": changes}

    @staticmethod
    def apply_delta(baseline: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """把增量应用到基线状态上，还原出完整状态。"""
        merged = dict(baseline)
        merged.update(changes)
        return merged

    # ------------------------------------------------------------------
    # 3. 插值：考虑延迟的平滑插值 + 有上限的外推
    # ------------------------------------------------------------------
    def update_latency(self, player_id: str, latency: float):
        self._latency[player_id] = max(0.0, float(latency))

    def interpolation(self, player_id: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """计算远端玩家在渲染时刻的平滑状态。

        渲染时刻 = now - 插值缓冲延迟 - 网络延迟；
        超出最新快照时按最近速度外推，外推时长受 extrapolation_limit 限制。
        """
        now = time.time() if now is None else now
        render_time = now - self.interpolation_delay - self._latency.get(player_id, 0.0)
        snaps = self._snapshots.get(player_id, [])
        if not snaps:
            return None
        if len(snaps) == 1 or render_time <= snaps[0][0]:
            return dict(snaps[0][1])
        for i in range(len(snaps) - 1):
            t0, s0 = snaps[i]
            t1, s1 = snaps[i + 1]
            if t0 <= render_time <= t1:
                alpha = 0.0 if t1 == t0 else (render_time - t0) / (t1 - t0)
                return {k: _lerp(s0.get(k), s1.get(k), alpha) for k in s1}
        # 丢包/延迟导致渲染时刻超出最新快照：按最近速度外推，但有上限
        t_last, s_last = snaps[-1]
        if len(snaps) >= 2:
            t_prev, s_prev = snaps[-2]
            dt = t_last - t_prev
            over = min(render_time - t_last, self.extrapolation_limit)
            if dt > 0 and over > 0:
                result = {}
                for key, value in s_last.items():
                    prev = s_prev.get(key)
                    if isinstance(value, (int, float)) and isinstance(prev, (int, float)):
                        result[key] = value + (value - prev) / dt * over
                    else:
                        result[key] = value
                return result
        return dict(s_last)

    # ------------------------------------------------------------------
    # 4. 丢包恢复：丢包检测 + 重传 + 状态合并
    # ------------------------------------------------------------------
    def receive_packet(self, player_id: str, seq: int, timestamp: float,
                       changes: Dict[str, Any], base_seq: Optional[int] = None) -> List[int]:
        """接收一个状态包，返回检测到的丢包序列号（NACK 列表）。

        状态合并按字段级 last-writer-wins：旧包不会覆盖更新的字段，
        但旧包中尚未更新的字段仍可补齐。
        """
        expected = self._recv_seq.get(player_id, 0) + 1
        missing: List[int] = []
        if seq > expected:
            missing = list(range(expected, seq))
            self._missing.setdefault(player_id, set()).update(missing)
        if seq > self._recv_seq.get(player_id, 0):
            self._recv_seq[player_id] = seq
        if player_id in self._missing:
            self._missing[player_id].discard(seq)

        field_seq = self._remote_field_seq.setdefault(player_id, {})
        state = self._remote_state.setdefault(player_id, {})
        for key, value in changes.items():
            if seq >= field_seq.get(key, -1):
                state[key] = value
                field_seq[key] = seq

        snaps = self._snapshots.setdefault(player_id, [])
        snaps.append((timestamp, dict(state)))
        snaps.sort(key=lambda s: s[0])
        if len(snaps) > _SNAPSHOT_BUFFER_SIZE:
            del snaps[:len(snaps) - _SNAPSHOT_BUFFER_SIZE]
        return missing

    def packet_loss_recovery(self, player_id: Optional[str] = None):
        """返回待重传的丢包序列号（NACK）。指定玩家返回列表，否则返回字典。"""
        if player_id is not None:
            return sorted(self._missing.get(player_id, set()))
        return {pid: sorted(seqs) for pid, seqs in self._missing.items() if seqs}

    def retransmit(self, requests: List[Tuple[str, int]]) -> List[Packet]:
        """根据 NACK 请求从历史中找出原包重传。"""
        resent = []
        for player_id, seq in requests:
            packet = self._history.get((player_id, seq))
            if packet is not None:
                resent.append(packet)
        return resent

    def acknowledge(self, player_id: str, seq: int):
        self._pending_acks.pop((player_id, seq), None)

    def get_remote_state(self, player_id: str) -> Dict[str, Any]:
        return dict(self._remote_state.get(player_id, {}))

    # ------------------------------------------------------------------
    # 5. 带宽优化：QoS 优先级调度 + 流量控制
    # ------------------------------------------------------------------
    def _flow_rate(self) -> float:
        return min(self.bandwidth_limit, self.receiver_capacity)

    def set_receiver_capacity(self, capacity: float):
        """接收方上报处理能力，发送方据此做流量控制。"""
        self.receiver_capacity = max(0.0, float(capacity))

    def _refill_tokens(self, now: float):
        rate = self._flow_rate()
        if self._token_time is None:
            self._token_time = now
            return
        elapsed = max(0.0, now - self._token_time)
        self._tokens = min(rate, self._tokens + rate * elapsed)
        self._token_time = now

    def bandwidth_optimization(self, packets: List[Packet],
                               now: Optional[float] = None) -> Dict[str, List[Packet]]:
        """QoS 调度 + 流量控制：高优先级包先发，发送量受令牌桶限制。"""
        now = time.time() if now is None else now
        self._refill_tokens(now)
        ordered = sorted(packets, key=lambda p: (-p.priority, p.seq))
        send: List[Packet] = []
        deferred: List[Packet] = []
        for packet in ordered:
            if packet.size <= self._tokens:
                self._tokens -= packet.size
                send.append(packet)
            else:
                deferred.append(packet)
        return {"send": send, "deferred": deferred}
