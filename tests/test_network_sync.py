"""
网络同步与状态压缩系统 - 验收测试
运行方式：python -m pytest tests/test_network_sync.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.network_sync import (
    NetworkSync,
    Packet,
    PRIORITY_HIGH,
    PRIORITY_MEDIUM,
    PRIORITY_LOW,
)


class TestNetworkSync:
    def setup_method(self):
        self.system = NetworkSync()

    # ---------------- 状态同步：增量 + 优先级 + 频率控制 ----------------

    def test_case_01(self):
        """增量同步：只发送发生变化的玩家，无变化的玩家不重复发送。"""
        sync = self.system
        sync.update_player_state("p1", {"x": 0, "y": 0})
        sync.update_player_state("p2", {"x": 5, "y": 5})
        packets = sync.state_synchronization(now=0.0)
        assert len(packets) == 2

        assert sync.state_synchronization(now=1.0) == []

        sync.update_player_state("p1", {"x": 10})
        packets = sync.state_synchronization(now=2.0)
        assert len(packets) == 1
        assert packets[0].player_id == "p1"

    def test_case_02(self):
        """优先级：高优先级状态的同步频率显著高于低优先级状态。"""
        sync = self.system
        high_count = 0
        low_count = 0
        for frame in range(200):
            now = frame * 0.01  # 100fps
            sync.update_player_state("fast", {"x": frame}, priority=PRIORITY_HIGH)
            sync.update_player_state("slow", {"x": frame}, priority=PRIORITY_LOW)
            for packet in sync.state_synchronization(now=now):
                if packet.player_id == "fast":
                    high_count += 1
                else:
                    low_count += 1
        assert high_count > low_count
        assert high_count >= 30   # 20Hz * 2s
        assert low_count <= 6     # 2Hz * 2s

    def test_case_03(self):
        """频率控制：每帧调用也不会每帧发包，同步频率受配置限制。"""
        sync = self.system
        total = 0
        for frame in range(240):  # 240fps，模拟1秒
            now = frame / 240.0
            sync.update_player_state("p", {"x": now})
            total += len(sync.state_synchronization(now=now))
        assert 9 <= total <= 11   # 默认 MEDIUM 10Hz

    # ---------------- 增量压缩：字段级差异，只发变化量 ----------------

    def test_case_04(self):
        """增量压缩只包含发生变化的字段。"""
        sync = self.system
        sync.update_player_state("p", {"x": 0, "y": 0, "hp": 100})
        sync.state_synchronization(now=0.0)
        sync.update_player_state("p", {"x": 5})
        delta = sync.delta_compression("p")
        assert delta["changes"] == {"x": 5}

    def test_case_05(self):
        """没有变化时增量为空，不发送任何数据。"""
        sync = self.system
        sync.update_player_state("p", {"x": 1, "y": 2})
        sync.state_synchronization(now=0.0)
        delta = sync.delta_compression("p")
        assert delta["changes"] == {}
        assert sync.state_synchronization(now=1.0) == []

    def test_case_06(self):
        """字段级增量应用到基线后能还原出完整状态。"""
        sync = self.system
        sync.update_player_state("p", {"x": 0, "y": 0, "hp": 100})
        sync.state_synchronization(now=0.0)
        sync.update_player_state("p", {"x": 7, "hp": 80})
        delta = sync.delta_compression("p")
        restored = NetworkSync.apply_delta({"x": 0, "y": 0, "hp": 100}, delta["changes"])
        assert restored == {"x": 7, "y": 0, "hp": 80}

    # ---------------- 插值：平滑 + 考虑延迟 + 外推 ----------------

    def test_case_07(self):
        """快照之间插值，远端玩家移动平滑而不是跳变。"""
        sync = self.system
        sync.receive_packet("p", seq=1, timestamp=0.0, changes={"x": 0.0})
        sync.receive_packet("p", seq=2, timestamp=0.1, changes={"x": 10.0})
        mid = sync.interpolation("p", now=0.15)  # 渲染时刻 = 0.15 - 0.1 = 0.05
        assert mid["x"] == pytest.approx(5.0)

    def test_case_08(self):
        """插值考虑网络延迟：延迟越高，渲染的状态越旧。"""
        def build(with_latency):
            sync = NetworkSync({"interpolation_delay": 0.1})
            sync.receive_packet("p", seq=1, timestamp=0.0, changes={"x": 0.0})
            sync.receive_packet("p", seq=2, timestamp=0.1, changes={"x": 10.0})
            sync.receive_packet("p", seq=3, timestamp=0.2, changes={"x": 20.0})
            if with_latency:
                sync.update_latency("p", 0.05)
            return sync.interpolation("p", now=0.25)  # 有延迟渲染0.1，无延迟渲染0.15

        delayed = build(True)
        normal = build(False)
        assert delayed["x"] == pytest.approx(10.0)
        assert normal["x"] == pytest.approx(15.0)
        assert delayed["x"] < normal["x"]

    def test_case_09(self):
        """丢包后按速度外推，玩家不卡在原地；外推有上限不会无限飞行。"""
        sync = self.system
        sync.receive_packet("p", seq=1, timestamp=0.0, changes={"x": 0.0})
        sync.receive_packet("p", seq=2, timestamp=0.1, changes={"x": 10.0})
        pos = sync.interpolation("p", now=0.3)  # 渲染时刻0.2，超出最新快照0.1
        assert pos["x"] > 10.0
        assert pos["x"] == pytest.approx(20.0)  # 速度100/s * 外推0.1

        far = sync.interpolation("p", now=100.0)
        assert far["x"] == pytest.approx(10.0 + 100.0 * 0.25)  # 外推上限0.25s

    # ---------------- 丢包恢复：检测 + 重传 + 状态合并 ----------------

    def test_case_10(self):
        """序列号不连续时能检测出丢失的包并要求重传。"""
        sync = self.system
        sync.receive_packet("p", seq=1, timestamp=0.0, changes={"x": 0})
        missing = sync.receive_packet("p", seq=3, timestamp=0.2, changes={"x": 20})
        assert missing == [2]
        assert sync.packet_loss_recovery("p") == [2]

    def test_case_11(self):
        """发送方重传丢失的包，接收方补齐后状态恢复。"""
        sender = self.system
        sender.update_player_state("p", {"x": 1})
        p1 = sender.state_synchronization(now=0.0)
        sender.update_player_state("p", {"x": 2})
        p2 = sender.state_synchronization(now=0.11)
        sender.update_player_state("p", {"x": 3})
        p3 = sender.state_synchronization(now=0.22)

        receiver = NetworkSync()
        receiver.receive_packet("p", p1[0].seq, p1[0].timestamp, p1[0].changes)
        missing = receiver.receive_packet("p", p3[0].seq, p3[0].timestamp, p3[0].changes)
        assert missing == [2]

        resent = sender.retransmit([("p", 2)])
        assert len(resent) == 1
        assert resent[0].changes == {"x": 2}
        receiver.receive_packet("p", resent[0].seq, resent[0].timestamp, resent[0].changes)
        assert receiver.packet_loss_recovery("p") == []
        assert receiver.get_remote_state("p")["x"] == 3

    def test_case_12(self):
        """状态合并：迟到的旧包不会覆盖更新的字段，但可补齐缺失字段。"""
        sync = self.system
        sync.receive_packet("p", seq=1, timestamp=0.0, changes={"x": 0, "y": 0})
        sync.receive_packet("p", seq=3, timestamp=0.2, changes={"x": 100})
        sync.receive_packet("p", seq=2, timestamp=0.1, changes={"x": 50, "y": 7})
        state = sync.get_remote_state("p")
        assert state["x"] == 100  # seq2 的旧值不能覆盖 seq3
        assert state["y"] == 7    # seq2 的新字段仍然合并进来

    # ---------------- 带宽优化：QoS + 流量控制 ----------------

    def test_case_13(self):
        """QoS：带宽不足时高优先级包先发，低优先级包延后。"""
        sync = NetworkSync({"bandwidth_limit": 100, "receiver_capacity": 100})
        high = Packet(player_id="h", seq=1, timestamp=0.0, base_seq=0,
                      changes={"x": 1}, priority=PRIORITY_HIGH, size=80)
        low = Packet(player_id="l", seq=1, timestamp=0.0, base_seq=0,
                     changes={"x": 1}, priority=PRIORITY_LOW, size=80)
        result = sync.bandwidth_optimization([low, high], now=0.0)
        assert result["send"] == [high]
        assert result["deferred"] == [low]

    def test_case_14(self):
        """流量控制：发送速率受接收方处理能力限制，令牌随时间补充。"""
        sync = NetworkSync({"bandwidth_limit": 1000, "receiver_capacity": 500})
        packets = [Packet(player_id="p", seq=i, timestamp=0.0, base_seq=0,
                          changes={}, priority=PRIORITY_MEDIUM, size=100)
                   for i in range(20)]
        first = sync.bandwidth_optimization(packets, now=0.0)
        first_bytes = sum(p.size for p in first["send"])
        assert first_bytes <= 500  # 初始令牌被接收方能力限制为500
        assert first["deferred"]

        second = sync.bandwidth_optimization(first["deferred"], now=1.0)
        second_bytes = sum(p.size for p in second["send"])
        assert second_bytes <= 500
        assert first_bytes + second_bytes <= 1000

    # ---------------- 端到端：100玩家场景带宽 ----------------

    def test_case_15(self):
        """端到端：100玩家首次全量后，单玩家移动的增量同步带宽远小于全量。"""
        sync = self.system
        for i in range(100):
            sync.update_player_state(
                f"p{i}",
                {"x": i, "y": i, "hp": 100, "mp": 50, "name": f"player{i}"},
            )
        full = sync.state_synchronization(now=0.0)
        full_bytes = sum(p.size for p in full)
        assert len(full) == 100

        sync.update_player_state("p0", {"x": 999})
        delta_packets = sync.state_synchronization(now=1.0)
        delta_bytes = sum(p.size for p in delta_packets)
        assert len(delta_packets) == 1
        assert delta_bytes < full_bytes * 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
