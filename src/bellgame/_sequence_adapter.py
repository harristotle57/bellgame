"""Private glue between bellgame's network dict and SeQUeNCe's router stack.

SeQUeNCe does the routing, Barrett-Kok entanglement generation, and swapping,
and tells us *when* things happen. bellgame works out *how good* the
entanglement is from the Fock model (see ``paths.py``).

We record, for every memory, when it last got entangled by a generation
protocol. Then we know how long each qubit sat in memory before it was used:

* at the two end nodes: from generation until the end-to-end pair is delivered
* at the middle nodes: from generation until the swap that consumed it
"""

import copy

from sequence.app.request_app import RequestApp
from sequence.entanglement_management.generation import EntanglementGenerationA
from sequence.entanglement_management.swapping import EntanglementSwappingA
from sequence.topology.router_net_topo import RouterNetTopo

PS_PER_MS = 1e9
TINY_FIDELITY = 1e-6  # accept every delivered pair; we compute quality ourselves


def sequence_config(net, seed=None):
    """A deep copy of the network dict with bellgame-only fields stripped, ready for SeQUeNCe."""
    config = copy.deepcopy(net)
    for q in config["qconnections"]:
        q.pop("photonics", None)
    if seed is not None:
        for i, node in enumerate(config["nodes"]):
            node["seed"] = int(seed) * 1000 + i
        for i, q in enumerate(config["qconnections"]):
            q["seed"] = int(seed) * 1000 + 500 + i
    return config


def build_topology(net, seed=None, stop_time_ps=None):
    """SeQUeNCe RouterNetTopo for this network."""
    config = sequence_config(net, seed)
    if stop_time_ps is not None:
        config["stop_time"] = int(stop_time_ps)
    return RouterNetTopo(config)


class _TimedRequestApp(RequestApp):
    """RequestApp that also notes how long the delivered end qubit waited in memory."""

    def __init__(self, node, recorder):
        super().__init__(node)
        self.recorder = recorder
        self.delivered_waits_ps = []
        self.delivery_times_ps = []

    def get_memory(self, info):
        if info.state == "ENTANGLED" and info.index in self.memo_to_reservation:
            reservation = self.memo_to_reservation[info.index]
            partner = reservation.responder if self.node.name == reservation.initiator else reservation.initiator
            if info.remote_node == partner:
                now = self.node.timeline.now()
                born = self.recorder.generated.get((self.node.name, info.memory.name), now)
                self.delivered_waits_ps.append(now - born)
                self.delivery_times_ps.append(now)
        super().get_memory(info)


class _Recorder:
    """Watches every router's resource manager for generation and swapping events."""

    def __init__(self):
        self.generated = {}       # (node, memory name) -> time of last generation (ps)
        self.swap_waits_ps = {}   # node -> list of storage times of memories consumed by swaps

    def watch(self, router):
        original_update = router.resource_manager.update
        timeline = router.timeline
        name = router.name

        def update(protocol, memory, state):
            now = timeline.now()
            if isinstance(protocol, EntanglementGenerationA) and state == "ENTANGLED":
                self.generated[(name, memory.name)] = now
            elif isinstance(protocol, EntanglementSwappingA):
                born = self.generated.get((name, memory.name))
                if born is not None:
                    self.swap_waits_ps.setdefault(name, []).append(now - born)
            original_update(protocol, memory, state)

        router.resource_manager.update = update


def run_request(net, src, dst, sim_time_s, seed=None):
    """Run SeQUeNCe: ``src`` asks for entanglement with ``dst`` for ``sim_time_s`` seconds.

    Returns a dict with the routed path, delivery count, and memory wait times (ms).
    """
    start_ps = int(1e9)  # 1 ms, leave time for the reservation messages
    end_ps = start_ps + int(sim_time_s * 1e12)
    topo = build_topology(net, seed=seed, stop_time_ps=end_ps + int(1e9))
    routers = {r.name: r for r in topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)}
    for n in (src, dst):
        if n not in routers:
            raise ValueError(f"no node called '{n}' in the network")
    recorder = _Recorder()
    for router in routers.values():
        recorder.watch(router)
    app_src = _TimedRequestApp(routers[src], recorder)
    app_dst = _TimedRequestApp(routers[dst], recorder)

    sizes = [n["memo_size"] for n in net["nodes"]]
    memo_size = max(1, min(sizes) // 2)
    app_src.start(dst, start_ps, end_ps, memo_size, TINY_FIDELITY)
    tl = topo.get_timeline()
    tl.init()
    tl.run()

    return {
        "path": list(app_src.path),
        "reserved": bool(app_src.reservation_result),
        "pairs": len(app_src.delivered_waits_ps),
        "src_waits_ms": [w / PS_PER_MS for w in app_src.delivered_waits_ps],
        "dst_waits_ms": [w / PS_PER_MS for w in app_dst.delivered_waits_ps],
        "swap_waits_ms": {k: [w / PS_PER_MS for w in v] for k, v in recorder.swap_waits_ps.items()},
        "sim_time_s": sim_time_s,
        "topology": topo,
    }
