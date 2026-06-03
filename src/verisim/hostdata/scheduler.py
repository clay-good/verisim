"""The concurrency scheduler + interleaving-entropy dial (SPEC-6 §3.2-3.3, §3.4; HC7, the H14 axis).

Concurrency is the host world's defining differentiator (§1.1) and the one nondeterminism source the
record/replay literature calls unsolved (HW-1). SPEC-6 does not pretend to solve it; it makes it a
**measured dial** (H14): under chaos-mode scheduling, faithful horizon should degrade with
**interleaving entropy**, and a recorded schedule should recover it.

This module is the dependency-free realization of that dial. A **workload** is a set of independent
*threads*, each a fixed little program ``fork → open → write → close → exit`` over an assigned path;
some paths are **shared** across threads, so the final file content is last-writer-wins and depends
on the interleaving (the fs subsystem becomes order-sensitive), and forks are interleaved so the
pid a thread is allocated depends on the global fork order (the proc subsystem becomes
order-sensitive). The :class:`HostScheduler` interleaves the threads into a single concrete syscall
sequence, parameterized by ``interleave`` ∈ [0, 1] (the chaos knob) and a ``chaos_seed``:

  - ``interleave = 0`` -- run each thread to completion before the next (near-sequential, the
    single-threaded floor regime, §3.3);
  - ``interleave = 1`` -- pick a uniformly random ready thread every step (maximal interleaving,
    chaos mode);
  - between -- with probability ``interleave`` switch to a random ready thread, else continue the
    current one.

The scheduler resolves each op against a live reference oracle (so a thread's actual pid/fd are the
ones the oracle will assign), and emits **concrete** :class:`~verisim.host.action.HostAction` s, so
replaying the sequence from the boot state reproduces the same trajectory bit-for-bit (the §3.3
determinism contract: deterministic given ``(snapshot, sched_seed)``). The *realized* interleaving
entropy is then read off the emitted schedule as the **context-switch rate**
(:func:`interleaving_entropy`) -- the measured independent variable H14 plots ``H_ε`` against.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass

from verisim.host.action import HostAction, parse_host_action
from verisim.host.config import HostConfig
from verisim.host.state import HostState
from verisim.hostoracle.reference import ReferenceHostOracle

# Each worker thread runs this fixed program; the scheduler interleaves threads at the op level.
_PROGRAM = ("fork", "open", "write", "close", "exit")


@dataclass
class Thread:
    """One logical process: a fixed program over ``path``, plus the runtime pid/fd it is given."""

    path: str
    token: str
    pid: int | None = None  # actual pid, assigned when its ``fork`` executes (order-dependent)
    fd: int | None = None  # actual fd, assigned when its ``open`` executes
    pc: int = 0  # program counter into ``_PROGRAM``

    @property
    def done(self) -> bool:
        return self.pc >= len(_PROGRAM)


def make_workload(config: HostConfig, n_threads: int, seed: int) -> list[Thread]:
    """Build ``n_threads`` worker threads with assigned paths/tokens; some paths are shared.

    Paths are drawn from a small prefix of the config pool so threads **collide** on files -- the
    shared-file contention that makes the fs subsystem order-sensitive. Deterministic in ``seed``.
    """
    rng = random.Random(seed)
    n_paths = max(1, min(len(config.paths), (n_threads + 1) // 2))  # ~2 threads per path -> sharing
    paths = config.paths[:n_paths]
    return [
        Thread(path=rng.choice(paths), token=rng.choice(config.content_tokens))
        for _ in range(n_threads)
    ]


@dataclass(frozen=True)
class Schedule:
    """One interleaved schedule: the concrete action sequence + the thread each action belongs to.

    ``actions`` replays from the boot state to a fixed trajectory; ``thread_ids`` is the per-step
    workload-thread index (so :func:`interleaving_entropy` measures *thread* switches, not the raw
    acting pid -- every ``fork``'s acting pid is the parent, which would confound the latter).
    """

    actions: list[HostAction]
    thread_ids: list[int]


@dataclass
class HostScheduler:
    """Interleaves a workload into one concrete syscall sequence under a chaos knob (§3.3)."""

    config: HostConfig
    interleave: float = 0.0  # the chaos dial: 0 == near-sequential, 1 == maximal interleaving

    def schedule(self, workload: Sequence[Thread], chaos_seed: int) -> Schedule:
        """Emit the interleaved concrete schedule (replayable from the boot state)."""
        rng = random.Random(chaos_seed)
        oracle = ReferenceHostOracle()
        state = HostState.initial()
        threads = [
            Thread(path=t.path, token=t.token) for t in workload
        ]  # fresh runtime copies
        actions: list[HostAction] = []
        thread_ids: list[int] = []
        last = -1
        while not all(t.done for t in threads):
            ready = [i for i, t in enumerate(threads) if not t.done]
            idx = self._pick(ready, last, rng)
            action, state = self._run_op(threads[idx], state, oracle)
            actions.append(action)
            thread_ids.append(idx)
            last = idx
        return Schedule(actions=actions, thread_ids=thread_ids)

    def _pick(self, ready: list[int], last: int, rng: random.Random) -> int:
        """Choose the next thread to advance per the chaos knob."""
        if last in ready and rng.random() >= self.interleave:
            return last  # keep running the current thread (low-interleave behavior)
        return rng.choice(ready)

    def _run_op(
        self, thread: Thread, state: HostState, oracle: ReferenceHostOracle
    ) -> tuple[HostAction, HostState]:
        """Translate the thread's next op to a concrete action, step the oracle, advance it."""
        op = _PROGRAM[thread.pc]
        if op == "fork":
            thread.pid = state.next_pid  # the pid the oracle's fork will allocate
            action = parse_host_action("fork 1")
        elif op == "open":
            action = parse_host_action(f"open {thread.pid} {thread.path}")
        elif op == "write":
            action = parse_host_action(f"write {thread.pid} {thread.fd} {thread.token}")
        elif op == "close":
            action = parse_host_action(f"close {thread.pid} {thread.fd}")
        else:  # exit
            action = parse_host_action(f"exit {thread.pid} 0")
        result = oracle.step(state, action)
        if op == "open":  # the oracle returns the allocated fd as stdout
            thread.fd = int(result.stdout)
        thread.pc += 1
        return action, result.state


def interleaving_entropy(thread_ids: Sequence[int]) -> float:
    """The realized interleaving entropy: the thread **context-switch rate** of the schedule (§3.4).

    The fraction of consecutive steps that switch to a *different thread* -- minimal for a
    sequential schedule (each thread runs to completion, switching only at thread boundaries),
    approaching ``1`` as the scheduler ping-pongs between threads every step. This is the measured
    independent variable H14 plots ``H_ε`` against; a property of the emitted schedule, not of the
    chaos knob (which only biases it).
    """
    if len(thread_ids) < 2:
        return 0.0
    switches = sum(1 for a, b in itertools.pairwise(thread_ids) if a != b)
    return switches / (len(thread_ids) - 1)
