"""
Deep Research Agent — core orchestrator (M1: Multi-Agent Orchestrator)

An async task-orchestration engine driven by a 9-state state machine:
  IDLE → PLANNING → DISPATCHING → COLLECTING → SYNTHESIZING → ADVERSARIAL → DONE
  On failure it enters REPLANNING, and may finally reach FAILED.

Highlights:
  - hand-written asyncio + DAG executor, no dependency on LangGraph/AutoGen
  - after topological sorting, tasks run concurrently layer by layer, with a Semaphore capping concurrency
  - three-level degradation: single-task timeout -> mark and continue; >50% failures -> re-plan; global timeout -> force synthesis
  - the state machine is a dict mapping, which makes adding states and transitions easy
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from .schemas import (
    OrchestratorState,
    SubTask,
    AgentResult,
    AgentStatus,
    ResearchReport,
    RunConfig,
    TaskType,
)
from .agent_pool import AgentPool
from ..planner.dag import DAG
from ..planner.planner import Planner, PlanParseError
from ..planner.budget_tracker import BudgetTracker
from ..utils.tracing import trace_chain

# M4: Memory Store type hints (lazy import to avoid a circular dependency)
SharedMemoryStore = Any


__all__ = ["Orchestrator"]


class Orchestrator:
    """Core orchestrator of the Deep Research Agent.

    Attributes:
        planner: adaptive planner, responsible for initial planning and incremental re-planning.
        agent_pool: agent object pool, managing the lifecycle of worker agents.
        budget_tracker: token budget tracker.
        memory_store: global shared memory holding all sub-task results and intermediate context.
        compressor: (reserved) context compressor interface.
    """

    def __init__(
        self,
        planner: Planner,
        agent_pool: AgentPool,
        budget_tracker: BudgetTracker | None = None,
        compressor: Any | None = None,
        adversarial_loop: Any | None = None,
        memory_store: Any | None = None,
        summarizer_policy: Any | None = None,
        summarizer_factory: Callable[[Any, list], Any] | None = None,
    ) -> None:
        self.planner = planner
        self.agent_pool = agent_pool
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.compressor = compressor
        self.adversarial_loop = adversarial_loop
        self.memory_store = memory_store
        self.summarizer_policy = summarizer_policy
        # (policy, tools) -> SummarizerAgent; the finance scenario injects a summarizer that carries the evidence ledger
        self.summarizer_factory = summarizer_factory

        # runtime state (the dict is kept as a fast cache; M4 provides persistence + semantic retrieval)
        self._memory_store: dict[str, Any] = {}
        self._results: list[AgentResult] = []
        self._dag: DAG | None = None
        self._task_map: dict[str, SubTask] = {}
        self._current_state = OrchestratorState.IDLE
        self._query: str = ""
        self._config: RunConfig = RunConfig()
        self._start_time: float = 0.0
        self._replan_count: int = 0
        self._adversarial_count: int = 0

        # state handler mapping
        self._state_handlers: dict[OrchestratorState, Callable[[], asyncio.Future[OrchestratorState]]] = {
            OrchestratorState.IDLE: self._on_idle,
            OrchestratorState.PLANNING: self._do_planning,
            OrchestratorState.DISPATCHING: self._do_dispatching,
            OrchestratorState.COLLECTING: self._do_collecting,
            OrchestratorState.SYNTHESIZING: self._do_synthesizing,
            OrchestratorState.ADVERSARIAL: self._do_adversarial,
            OrchestratorState.REPLANNING: self._do_replanning,
            OrchestratorState.DONE: self._on_done,
            OrchestratorState.FAILED: self._on_failed,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @trace_chain(name="orchestrator.run", tags=["m1", "orchestrator"])
    async def run(self, query: str, config: RunConfig | None = None) -> ResearchReport:
        """Main entry point: run the complete research flow.

        Args:
            query: the research question.
            config: run configuration, defaults to RunConfig().

        Returns:
            ResearchReport: the final research report.
        """
        self._query = query
        self._config = config or RunConfig()
        self._start_time = time.monotonic()
        self._replan_count = 0
        self._adversarial_count = 0
        self._memory_store.clear()
        self._results.clear()
        self._dag = None
        self._task_map.clear()
        self._current_state = OrchestratorState.IDLE

        # state machine main loop
        while self._current_state not in (OrchestratorState.DONE, OrchestratorState.FAILED):
            # global timeout check
            if self._is_global_timeout():
                if self._current_state in (
                    OrchestratorState.COLLECTING,
                    OrchestratorState.SYNTHESIZING,
                    OrchestratorState.ADVERSARIAL,
                ):
                    # force synthesis: generate the report from the existing results
                    self._current_state = OrchestratorState.SYNTHESIZING
                else:
                    self._current_state = OrchestratorState.FAILED
                break

            handler = self._state_handlers.get(self._current_state)
            if handler is None:
                raise RuntimeError(f"Unknown state: {self._current_state}")

            next_state = await handler()
            self._current_state = next_state

            print(f"[Orchestrator] State transition: {self._current_state.value}")

        # return the result
        if self._current_state == OrchestratorState.DONE:
            # the final report should be in memory
            report = self._memory_store.get("final_report")
            if report is None:
                report = ResearchReport(query=query, content="Report generation failed unexpectedly.")
            report.num_replan = self._replan_count
            report.adversarial_rounds = self._adversarial_count

            # M4: store the final report in the SharedMemoryStore
            if self.memory_store is not None:
                try:
                    from src.memory.long_term import MemoryEntry
                    entry = MemoryEntry(
                        entry_id=f"final_report:{int(time.time())}",
                        claim=str(report.content)[:800],
                        source="orchestrator",
                        confidence=report.confidence,
                        agent_id="orchestrator",
                        timestamp=time.time(),
                        evidence_type="primary",
                        embedding=[],
                        topic=query[:50],
                        metadata={
                            "num_searches": report.num_searches,
                            "num_replan": report.num_replan,
                            "adversarial_rounds": report.adversarial_rounds,
                        },
                    )
                    self.memory_store.put(entry)
                    print(f"[M4] Final report stored to memory (confidence={report.confidence:.2f})")
                except Exception as e:
                    print(f"[M4] Failed to store final report: {e}")

            return report

        # FAILED state
        return ResearchReport(
            query=query,
            content="Research failed due to persistent errors or global timeout.",
            num_replan=self._replan_count,
            adversarial_rounds=self._adversarial_count,
        )

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    async def _on_idle(self) -> OrchestratorState:
        """Automatically move from IDLE to PLANNING."""
        return OrchestratorState.PLANNING

    async def _do_planning(self) -> OrchestratorState:
        """Call the Planner to generate the initial DAG.

        On failure go straight to FAILED (a failed initial plan cannot be recovered).
        """
        try:
            memory_ctx = self._build_memory_context()
            self._dag = self.planner.generate_plan(self._query, memory_ctx)
            # get the full SubTask info (description, search_hints, etc.) from the planner
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                # degrade: if parsing fails, use placeholders
                self._task_map = self._rebuild_task_map_from_dag()
        except PlanParseError as e:
            print(f"[Planning] Failed: {e}")
            return OrchestratorState.FAILED
        except Exception as e:
            print(f"[Planning] Unexpected error: {e}")
            return OrchestratorState.FAILED

        n_tasks = len(self._dag)
        n_layers = len(self._dag.get_parallel_groups()) if self._dag else 0
        print(f"[Planning] ✓ DAG generated: {n_tasks} sub-tasks, {n_layers} execution layers")
        # print the sub-task descriptions for diagnosis
        for tid, task in self._task_map.items():
            print(f"[Planning]   {tid}: {task.description}")
        return OrchestratorState.DISPATCHING

    async def _do_dispatching(self) -> OrchestratorState:
        """Topological sort + concurrent scheduling of sub-agents.

        Core logic:
          1. get the parallel execution layers (parallel groups)
          2. within each layer run concurrently with asyncio.gather + Semaphore
          3. each sub-task gets its own timeout (asyncio.wait_for)
          4. collect the results into self._results
        """
        if self._dag is None or len(self._dag) == 0:
            return OrchestratorState.COLLECTING

        semaphore = asyncio.Semaphore(self._config.max_concurrent)
        parallel_groups = self._dag.get_parallel_groups()
        all_results: list[AgentResult] = []

        for layer_idx, group in enumerate(parallel_groups):
            print(f"[Dispatch] ▶ Layer {layer_idx + 1}/{len(parallel_groups)}: {group} (parallel)")

            # build the coroutine list for this layer
            async def _run_one(task_id: str) -> AgentResult:
                async with semaphore:
                    subtask = self._task_map.get(task_id)
                    if subtask is None:
                        return AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"SubTask '{task_id}' not found in task_map",
                        )

                    # prepare the context: first the results of the dependency tasks
                    context = self._build_task_context(subtask)

                    # get an Agent
                    agent = await self.agent_pool.get_agent(subtask.task_type)
                    try:
                        # set the single-task timeout
                        result = await asyncio.wait_for(
                            agent.run(subtask, context),
                            timeout=subtask.timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        result = AgentResult(
                            task_id=task_id,
                            status=AgentStatus.TIMEOUT,
                            output=f"Task timed out after {subtask.timeout_seconds}s",
                        )
                    except Exception as e:
                        result = AgentResult(
                            task_id=task_id,
                            status=AgentStatus.FAILED,
                            output=f"Exception: {type(e).__name__}: {e}",
                        )
                    finally:
                        await self.agent_pool.release_agent(agent)

                    return result

            # run this layer concurrently
            coros = [_run_one(tid) for tid in group]
            layer_results = await asyncio.gather(*coros, return_exceptions=True)

            for lr in layer_results:
                if isinstance(lr, Exception):
                    # wrap the exception as a FAILED result
                    # in theory this cannot happen (_run_one already catches), but just to be safe
                    all_results.append(AgentResult(
                        task_id="unknown",
                        status=AgentStatus.FAILED,
                        output=f"Dispatch exception: {lr}",
                    ))
                else:
                    all_results.append(lr)

        self._results = all_results
        return OrchestratorState.COLLECTING

    async def _do_collecting(self) -> OrchestratorState:
        """Collect results, write them to memory, and check whether re-planning is needed.

        Checkpoints of the three-level degradation strategy:
          - single-task timeout / failure: already handled in the dispatch layer (mark the status, continue)
          - >50% failures: trigger REPLANNING
          - global timeout: handled by the loop check in the outer run()
        """
        # write the results into the runtime memory dict
        for r in self._results:
            self._memory_store[f"result:{r.task_id}"] = r

        # M4: sync successful results into the SharedMemoryStore (persistence + vector index)
        if self.memory_store is not None:
            for r in self._results:
                if r.status == AgentStatus.SUCCESS and r.output:
                    self._sync_result_to_memory_store(r)

        success_count = sum(1 for r in self._results if r.status == AgentStatus.SUCCESS)
        total_count = len(self._results)
        fail_count = total_count - success_count
        status_icon = "✓" if success_count == total_count else "⚠"
        print(f"[Collect] {status_icon} sub-tasks done: {success_count}/{total_count} succeeded", end="")
        if fail_count > 0:
            print(f" ({fail_count} failed)")
        else:
            print()

        # check whether re-planning is needed
        if self._should_replan(self._results):
            if self._replan_count < self._config.max_replan_rounds:
                self._replan_count += 1
                return OrchestratorState.REPLANNING
            else:
                print("[Collect] Max replan rounds reached, proceeding with partial results")
                # max re-plan count exceeded, continue to synthesis (with the existing results)

        return OrchestratorState.SYNTHESIZING

    def _sync_result_to_memory_store(self, result: AgentResult) -> None:
        """Sync an AgentResult into the M4 SharedMemoryStore.

        Extracts the key claim in the output as a memory entry, to support later semantic retrieval.
        """
        try:
            # lazy import to avoid a circular dependency
            from src.memory.long_term import MemoryEntry
            claim_text = str(result.output)[:500]  # take the first 500 chars as the claim
            entry = MemoryEntry(
                entry_id=result.task_id,
                claim=claim_text,
                source=f"task:{result.task_id}",
                confidence=getattr(result, "confidence", 0.5),
                agent_id=result.task_id,
                timestamp=time.time(),
                evidence_type="primary",
                embedding=[],  # SharedMemoryStore.put() generates the embedding automatically
                topic=self._query[:50],
                metadata={
                    "status": result.status.value,
                    "token_usage": getattr(result, "token_usage", 0),
                },
            )
            self.memory_store.put(entry)
            print(f"[M4] Memory stored: {result.task_id} (claim={claim_text[:60]}...)")
        except Exception as e:
            print(f"[M4] Failed to store memory for {result.task_id}: {e}")

    async def _do_synthesizing(self) -> OrchestratorState:
        """Call the SummarizerAgent to synthesize the research report."""
        # create the synthesis task
        synth_task = SubTask(
            task_id="synthesize_final",
            task_type=TaskType.ANALYZE,  # use the ANALYZE type; it is actually handled by the SummarizerAgent
            description="Synthesize all sub-task results into a final research report.",
            timeout_seconds=300,
        )

        context = {
            "query": self._query,
            "results": self._results,
        }

        agent = await self.agent_pool.get_agent(TaskType.ANALYZE)
        # a SummarizerAgent is needed, but agent_pool may return a ResearcherAgent
        # here we create a SummarizerAgent through a type check or by force
        from ..agents.summarizer import SummarizerAgent
        if not isinstance(agent, SummarizerAgent):
            # prefer the configured summarizer_policy (larger max_tokens), fall back to agent.policy
            policy = self.summarizer_policy or agent.policy
            if self.summarizer_factory is not None:
                agent = self.summarizer_factory(policy, agent.tools)
            else:
                agent = SummarizerAgent(name="summarizer", policy=policy, tools=agent.tools)

        try:
            result = await asyncio.wait_for(
                agent.run(synth_task, context),
                timeout=synth_task.timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = AgentResult(
                task_id="synthesize_final",
                status=AgentStatus.TIMEOUT,
                output="Synthesis timed out",
            )
        except Exception as e:
            result = AgentResult(
                task_id="synthesize_final",
                status=AgentStatus.FAILED,
                output=f"Synthesis error: {type(e).__name__}: {e}",
            )
        finally:
            await self.agent_pool.release_agent(agent)

        if result.status == AgentStatus.SUCCESS and isinstance(result.output, ResearchReport):
            self._memory_store["final_report"] = result.output
        else:
            # synthesis failed but results exist, generate a degraded report
            self._memory_store["final_report"] = ResearchReport(
                query=self._query,
                content=str(result.output) if result.output else "Synthesis failed.",
                confidence=0.0,
                num_searches=sum(
                    len([t for t in r.trajectory if t.get("role") == "tool"])
                    for r in self._results
                ),
            )

        if self._config.enable_adversarial:
            print("[Synthesize] ✓ report synthesized, entering adversarial optimization")
            return OrchestratorState.ADVERSARIAL
        print("[Synthesize] ✓ report synthesized")
        return OrchestratorState.DONE

    async def _do_adversarial(self) -> OrchestratorState:
        """M5: Red-Blue adversarial denoising loop.

        Calls the AdversarialLoop to iteratively challenge-and-verify the report.
        Triggered only when the report confidence is below the threshold, to avoid wasting resources.
        """
        report = self._memory_store.get("final_report")
        if report is None:
            return OrchestratorState.DONE

        # skip the adversarial step when confidence is already high
        if report.confidence >= 0.8:
            print("[Adversarial] ✓ report confidence meets the bar (≥0.8), skipping adversarial optimization")
            return OrchestratorState.DONE

        if self.adversarial_loop is None:
            print("[Adversarial] AdversarialLoop not configured, skipping")
            return OrchestratorState.DONE

        try:
            print(f"[Adversarial] ▶ starting Red-Blue adversarial optimization (current confidence={report.confidence:.2f})")
            optimized_report, history = await self.adversarial_loop.run(report)
            self._memory_store["final_report"] = optimized_report
            self._adversarial_count += len(history)
            print(f"[Adversarial] ✓ adversarial optimization done: {len(history)} rounds, final confidence={optimized_report.confidence:.2f}")
        except Exception as e:
            print(f"[Adversarial] ✗ adversarial optimization failed: {e}, using the original report")

        return OrchestratorState.DONE

    async def _do_replanning(self) -> OrchestratorState:
        """Trigger incremental re-planning.

        Keeps successful results with confidence≥0.6 and modifies the failed sub-questions.
        """
        failed_tasks = []
        for r in self._results:
            if r.status != AgentStatus.SUCCESS:
                st = self._task_map.get(r.task_id)
                if st:
                    failed_tasks.append(st)

        reason = self._build_failure_reason(self._results)
        print(f"[Replan] Round {self._replan_count}/{self._config.max_replan_rounds}. Failed tasks: {[t.task_id for t in failed_tasks]}")

        try:
            new_dag = self.planner.replan(
                query=self._query,
                failed_tasks=failed_tasks,
                existing_results=self._results,
                reason=reason,
            )
            self._dag = new_dag
            self._task_map = self.planner.get_task_map_from_dag(self._dag, self.planner._last_raw_json)
            if not self._task_map:
                self._task_map = self._rebuild_task_map_from_dag()
            # clear the previous round's results (kept in memory; new tasks can reference them via context_keys)
            self._results = []
        except PlanParseError as e:
            print(f"[Replan] Failed: {e}")
            # re-planning failed; if some results succeeded, try to synthesize directly
            if any(r.status == AgentStatus.SUCCESS for r in self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED
        except Exception as e:
            print(f"[Replan] Unexpected error: {e}")
            if any(r.status == AgentStatus.SUCCESS for r in self._results):
                return OrchestratorState.SYNTHESIZING
            return OrchestratorState.FAILED

        return OrchestratorState.DISPATCHING

    async def _on_done(self) -> OrchestratorState:
        """Terminal state, no further transitions."""
        return OrchestratorState.DONE

    async def _on_failed(self) -> OrchestratorState:
        """Terminal state, no further transitions."""
        return OrchestratorState.FAILED

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _should_replan(self, results: list[AgentResult]) -> bool:
        """Decide whether re-planning is needed.

        Policy:
          - triggered when the failure rate is > 50%
          - or when there is any TIMEOUT and fewer than 30% of results succeeded
        """
        if not results:
            return False
        total = len(results)
        failed = sum(1 for r in results if r.status in (AgentStatus.FAILED, AgentStatus.TIMEOUT))
        success = sum(1 for r in results if r.status == AgentStatus.SUCCESS)

        failure_rate = failed / total
        if failure_rate > 0.5:
            return True
        if success / total < 0.3 and failed > 0:
            return True
        return False

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _is_global_timeout(self) -> bool:
        """Check whether the global timeout has been exceeded."""
        elapsed = time.monotonic() - self._start_time
        return elapsed > self._config.global_timeout_seconds

    def _build_memory_context(self) -> str:
        """Build the context summary for the planner.

        Prefers semantic retrieval from the M4 SharedMemoryStore (if connected),
        otherwise falls back to iterating the runtime dict.
        """
        # M4: semantic retrieval of related memories
        if self.memory_store is not None:
            try:
                ctx = self.memory_store.get_context_for_query(
                    self._query, max_tokens=2000
                )
                if ctx:
                    print(f"[M4] Retrieved {len(ctx)} chars of semantic memory context")
                    return ctx
            except Exception as e:
                print(f"[M4] Semantic memory query failed: {e}, falling back to dict")

        # fallback: iterate the runtime dict
        parts = []
        for key, value in self._memory_store.items():
            if key.startswith("result:"):
                continue
            parts.append(f"{key}: {str(value)[:200]}")

        # M3: enable compression if the context is too long
        if self.compressor is not None and parts:
            total_chars = sum(len(p) for p in parts)
            if total_chars > 6000:  # heuristic threshold of about 2000 tokens
                try:
                    compressed = self.compressor.compress(
                        texts=parts,
                        query=self._query,
                        system_prompt_tokens=0,
                    )
                    print(f"[M3] Context compressed: {total_chars} → {sum(len(c) for c in compressed)} chars")
                    return "\n".join(compressed)
                except Exception as e:
                    print(f"[M3] Compression failed: {e}, using raw context")

        return "\n".join(parts) if parts else ""

    def _build_task_context(self, subtask: SubTask) -> dict:
        """Build the execution context for a single SubTask."""
        ctx = dict(self._memory_store)
        ctx["query"] = self._query
        # inject the results of the dependency tasks
        for dep_id in subtask.dependencies:
            dep_key = f"result:{dep_id}"
            if dep_key in self._memory_store:
                ctx[f"dep:{dep_id}"] = self._memory_store[dep_key]
        return ctx

    def _build_failure_reason(self, results: list[AgentResult]) -> str:
        """Analyze the failure reasons and produce a description for the replanner."""
        reasons = []
        timeout_count = sum(1 for r in results if r.status == AgentStatus.TIMEOUT)
        failed_count = sum(1 for r in results if r.status == AgentStatus.FAILED)
        if timeout_count > 0:
            reasons.append(f"{timeout_count} tasks timed out (may need simpler queries or longer timeout)")
        if failed_count > 0:
            reasons.append(f"{failed_count} tasks failed with errors")
        return "; ".join(reasons) if reasons else "Unknown failure"

    def _rebuild_task_map_from_dag(self) -> dict[str, SubTask]:
        """Rebuild the task_map from the DAG (using placeholders when the original SubTask info is missing).

        In real use the planner should return the full SubTask list;
        this is a fallback: create a default SubTask for each node in the DAG.
        """
        if self._dag is None:
            return {}

        task_map: dict[str, SubTask] = {}
        for node_id in self._dag:
            deps = self._dag.get_dependencies(node_id)
            if node_id not in self._task_map:
                # create a new placeholder SubTask
                task_map[node_id] = SubTask(
                    task_id=node_id,
                    task_type=TaskType.SEARCH,
                    description=f"Auto-generated task for {node_id}",
                    dependencies=deps,
                )
            else:
                # keep existing info, update the dependencies
                old = self._task_map[node_id]
                task_map[node_id] = SubTask(
                    task_id=old.task_id,
                    task_type=old.task_type,
                    description=old.description,
                    dependencies=deps,
                    context_keys=old.context_keys,
                    timeout_seconds=old.timeout_seconds,
                    priority=old.priority,
                    expected_type=old.expected_type,
                    search_hints=old.search_hints,
                )
        return task_map
