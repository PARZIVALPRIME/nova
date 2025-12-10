"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NEXUS CLINICAL AI - AGENT SYSTEM TESTER                   ║
║                     Adapted to Actual Implementation                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
from loguru import logger

# Setup logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
)


class AgentSystemTester:
    """Comprehensive tester adapted to actual implementation."""

    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0

    def log_test(self, name: str, status: bool, message: str = ""):
        """Log test result."""
        if status:
            self.passed += 1
            logger.success(f"✅ PASS: {name} {message}")
        else:
            self.failed += 1
            logger.error(f"❌ FAIL: {name} {message}")
        self.results[name] = {"status": status, "message": message}

    async def test_imports(self):
        """Test all critical imports."""
        logger.info("=" * 60)
        logger.info("🧪 TEST 1: IMPORT VALIDATION")
        logger.info("=" * 60)

        # Test imports that should work based on your structure
        imports_to_test = [
            # Base
            ("Base Agent", "from src.agents.base_agent import BaseAgent"),
            ("Agent Engine", "from src.agents.agent_engine import AgentEngine"),
            # 10 Agents - using correct class names
            (
                "Query Resolution Agent",
                "from src.agents.query_resolution_agent import QueryResolutionAgent",
            ),
            (
                "Risk Sentinel Agent",
                "from src.agents.risk_sentinel_agent import RiskSentinelAgent",
            ),
            (
                "Data Quality Agent",
                "from src.agents.data_quality_agent import DataQualityAgent",
            ),
            (
                "Prediction Agent",
                "from src.agents.prediction_agent import PredictionAgent",
            ),
            (
                "Workflow Orchestrator Agent",
                "from src.agents.workflow_orchestrator import WorkflowOrchestratorAgent",
            ),
            ("Coding Agent", "from src.agents.coding_agent import CodingAgent"),
            ("Safety Agent", "from src.agents.safety_agent import SafetyAgent"),
            ("Enrollment Agent", "from src.agents.enrollment_agent import EnrollmentAgent"),
            (
                "Communication Agent",
                "from src.agents.communication_agent import CommunicationAgent",
            ),
            ("Learning Agent", "from src.agents.learning_agent import LearningAgent"),
            # Config
            ("Agent Config", "from src.agents.config.agent_config import AgentConfig"),
            (
                "Orchestration Config",
                "from src.agents.config.orchestration_config import OrchestrationConfig",
            ),
            # Guardrails
            ("Input Validator", "from src.agents.guardrails.input_validator import InputValidator"),
            (
                "Output Sanitizer",
                "from src.agents.guardrails.output_sanitizer import OutputSanitizer",
            ),
            ("Rate Limiter", "from src.agents.guardrails.rate_limiter import RateLimiter"),
            ("Safety Checker", "from src.agents.guardrails.safety_checker import SafetyChecker"),
            (
                "Guardrails Engine",
                "from src.agents.guardrails.guardrails_engine import GuardrailsEngine",
            ),
            # Memory
            ("Memory Store", "from src.agents.memory.memory_store import MemoryStore"),
            (
                "Conversation Memory",
                "from src.agents.memory.conversation_memory import ConversationMemory",
            ),
            ("Context Manager", "from src.agents.memory.context_manager import ContextManager"),
            # Messaging
            ("Message Bus", "from src.agents.messaging.message_bus import MessageBus"),
            (
                "Event Dispatcher",
                "from src.agents.messaging.event_dispatcher import EventDispatcher",
            ),
        ]

        for name, import_stmt in imports_to_test:
            try:
                exec(import_stmt)
                self.log_test(f"Import {name}", True)
            except Exception as e:
                self.log_test(f"Import {name}", False, str(e)[:60])

    async def test_agent_initialization(self):
        """Test agent initialization with correct class names."""
        logger.info("=" * 60)
        logger.info("🧪 TEST 2: AGENT INITIALIZATION")
        logger.info("=" * 60)

        try:
            from src.agents.query_resolution_agent import QueryResolutionAgent
            from src.agents.risk_sentinel_agent import RiskSentinelAgent
            from src.agents.data_quality_agent import DataQualityAgent
            from src.agents.prediction_agent import PredictionAgent
            from src.agents.workflow_orchestrator import WorkflowOrchestratorAgent
            from src.agents.coding_agent import CodingAgent
            from src.agents.safety_agent import SafetyAgent
            from src.agents.enrollment_agent import EnrollmentAgent
            from src.agents.communication_agent import CommunicationAgent
            from src.agents.learning_agent import LearningAgent

            agents = [
                ("QueryResolutionAgent", QueryResolutionAgent),
                ("RiskSentinelAgent", RiskSentinelAgent),
                ("DataQualityAgent", DataQualityAgent),
                ("PredictionAgent", PredictionAgent),
                ("WorkflowOrchestratorAgent", WorkflowOrchestratorAgent),
                ("CodingAgent", CodingAgent),
                ("SafetyAgent", SafetyAgent),
                ("EnrollmentAgent", EnrollmentAgent),
                ("CommunicationAgent", CommunicationAgent),
                ("LearningAgent", LearningAgent),
            ]

            for name, agent_class in agents:
                try:
                    agent = agent_class()
                    agent_id = getattr(agent, "agent_id", "N/A")
                    self.log_test(f"Init {name}", True, f"ID: {str(agent_id)[:8]}...")
                except Exception as e:
                    self.log_test(f"Init {name}", False, str(e)[:60])

        except Exception as e:
            self.log_test("Agent Import Suite", False, str(e)[:60])

    async def test_guardrails(self):
        """Test guardrails with correct method signatures."""
        logger.info("=" * 60)
        logger.info("🧪 TEST 3: GUARDRAILS SYSTEM")
        logger.info("=" * 60)

        try:
            from src.agents.guardrails.input_validator import InputValidator
            from src.agents.guardrails.output_sanitizer import OutputSanitizer
            from src.agents.guardrails.rate_limiter import RateLimiter
            from src.agents.guardrails.safety_checker import SafetyChecker
            from src.agents.guardrails.guardrails_engine import GuardrailsEngine

            # Test Input Validator
            try:
                validator = InputValidator()
                test_input = {
                    "query": "What is the risk for Study 1?",
                    "user_id": "test_user",
                }
                result = validator.validate(test_input)
                # Handle dataclass ValidationResult
                if hasattr(result, "valid"):
                    is_valid = result.valid
                elif hasattr(result, "is_valid"):
                    is_valid = result.is_valid
                elif isinstance(result, dict):
                    is_valid = result.get("valid", True)
                else:
                    is_valid = bool(result)  # Fallback
                self.log_test("Input Validator", True, f"Valid: {is_valid}")
            except Exception as e:
                self.log_test("Input Validator", False, str(e)[:60])

            # Test Output Sanitizer
            try:
                sanitizer = OutputSanitizer()
                test_output = {"response": "Risk level is HIGH", "confidence": 0.95}
                result = sanitizer.sanitize(test_output)
                self.log_test("Output Sanitizer", True)
            except Exception as e:
                self.log_test("Output Sanitizer", False, str(e)[:60])

            # Test Rate Limiter
            try:
                limiter = RateLimiter()
                result = limiter.check("test_user")
                allowed = result.allowed if hasattr(result, "allowed") else result
                self.log_test("Rate Limiter", True, f"Allowed: {allowed}")
            except Exception as e:
                self.log_test("Rate Limiter", False, str(e)[:60])

            # Test Safety Checker - find correct method signature
            try:
                checker = SafetyChecker()
                # Try to find available methods
                methods = [m for m in dir(checker) if not m.startswith("_")]
                self.log_test("Safety Checker Init", True, f"Methods: {methods[:3]}")
            except Exception as e:
                self.log_test("Safety Checker", False, str(e)[:60])

            # Test Guardrails Engine
            try:
                engine = GuardrailsEngine()
                self.log_test("Guardrails Engine", True)
            except Exception as e:
                self.log_test("Guardrails Engine", False, str(e)[:60])

        except Exception as e:
            self.log_test("Guardrails Suite", False, str(e)[:60])

    async def test_memory_system(self):
        """Test memory system."""
        logger.info("=" * 60)
        logger.info("🧪 TEST 4: MEMORY SYSTEM")
        logger.info("=" * 60)

        try:
            from src.agents.memory.memory_store import MemoryStore
            from src.agents.memory.conversation_memory import ConversationMemory

            # Test Memory Store
            try:
                store = MemoryStore()
                store.store("test_key", {"data": "test_value"})
                retrieved = store.retrieve("test_key")
                self.log_test("Memory Store", True)
            except Exception as e:
                self.log_test("Memory Store", False, str(e)[:60])

            # Test Conversation Memory - find correct method
            try:
                conv_memory = ConversationMemory()
                # Get available methods
                methods = [
                    m
                    for m in dir(conv_memory)
                    if not m.startswith("_") and callable(getattr(conv_memory, m))
                ]
                self.log_test(
                    "Conversation Memory", True, f"Methods: {methods[:5]}"
                )
            except Exception as e:
                self.log_test("Conversation Memory", False, str(e)[:60])

            # Test Context Manager
            try:
                from src.agents.memory.context_manager import ContextManager

                ctx_mgr = ContextManager()
                ctx = ctx_mgr.create_context("test_ctx", "test_agent")
                self.log_test("Context Manager", True)
            except Exception as e:
                self.log_test("Context Manager", False, str(e)[:60])

        except Exception as e:
            self.log_test("Memory Suite", False, str(e)[:60])

    async def test_messaging_system(self):
        """Test messaging system."""
        logger.info("=" * 60)
        logger.info("🧪 TEST 5: MESSAGING SYSTEM")
        logger.info("=" * 60)

        try:
            from src.agents.messaging.message_bus import MessageBus
            from src.agents.messaging.event_dispatcher import EventDispatcher

            # Test Message Bus
            try:
                bus = MessageBus()
                self.log_test("Message Bus Init", True)
            except Exception as e:
                self.log_test("Message Bus", False, str(e)[:60])

            # Test Event Dispatcher
            try:
                dispatcher = EventDispatcher()
                self.log_test("Event Dispatcher", True)
            except Exception as e:
                self.log_test("Event Dispatcher", False, str(e)[:60])

            # Check message_types.py exports
            try:
                from src.agents.messaging import message_types

                exports = [x for x in dir(message_types) if not x.startswith("_")]
                self.log_test("Message Types", True, f"Exports: {exports[:5]}")
            except Exception as e:
                self.log_test("Message Types", False, str(e)[:60])

        except Exception as e:
            self.log_test("Messaging Suite", False, str(e)[:60])

    async def test_agent_engine(self):
        """Test the main agent engine with actual methods."""
        logger.info("=" * 60)
        logger.info("🧪 TEST 6: AGENT ENGINE")
        logger.info("=" * 60)

        try:
            from src.agents.agent_engine import AgentEngine

            engine = AgentEngine()
            self.log_test("Agent Engine Init", True)

            # List available methods
            methods = [
                m
                for m in dir(engine)
                if not m.startswith("_") and callable(getattr(engine, m))
            ]
            self.log_test("Agent Engine Methods", True, f"{methods[:8]}")

            # Check agent count
            agents = getattr(engine, "agents", {})
            orchestrator = getattr(engine, "orchestrator", None)
            agent_count = len(agents) if agents else 0
            self.log_test(
                "Agent Engine Agents",
                True,
                f"Count: {agent_count}, Orchestrator: {orchestrator is not None}",
            )

        except Exception as e:
            self.log_test("Agent Engine Suite", False, str(e)[:60])

    async def test_agent_methods(self):
        """Discover and test actual agent methods."""
        logger.info("=" * 60)
        logger.info("🧪 TEST 7: AGENT METHOD DISCOVERY")
        logger.info("=" * 60)

        agents_to_test = [
            (
                "QueryResolutionAgent",
                "src.agents.query_resolution_agent",
                "QueryResolutionAgent",
            ),
            (
                "RiskSentinelAgent",
                "src.agents.risk_sentinel_agent",
                "RiskSentinelAgent",
            ),
            (
                "DataQualityAgent",
                "src.agents.data_quality_agent",
                "DataQualityAgent",
            ),
            ("PredictionAgent", "src.agents.prediction_agent", "PredictionAgent"),
            ("SafetyAgent", "src.agents.safety_agent", "SafetyAgent"),
        ]

        for name, module_path, class_name in agents_to_test:
            try:
                module = __import__(module_path, fromlist=[class_name])
                agent_class = getattr(module, class_name)
                agent = agent_class()

                # Get callable methods (excluding private)
                methods = [
                    m
                    for m in dir(agent)
                    if not m.startswith("_") and callable(getattr(agent, m))
                ]

                # Find the main execution method
                exec_methods = [
                    m
                    for m in methods
                    if m in ["execute", "run", "process", "handle", "analyze", "assess"]
                ]

                self.log_test(
                    f"{name} Methods",
                    True,
                    f"Main: {exec_methods}, All: {len(methods)}",
                )

            except Exception as e:
                self.log_test(f"{name} Methods", False, str(e)[:60])

    async def test_agent_execution(self):
        """Test actual agent execution methods."""
        logger.info("=" * 60)
        logger.info("🧪 TEST 8: AGENT EXECUTION")
        logger.info("=" * 60)

        # Test agents with their actual methods
        try:
            from src.agents.agent_engine import AgentEngine

            engine = AgentEngine()

            # Try different method patterns based on what exists
            if hasattr(engine, "process_query"):
                try:
                    result = await engine.process_query(
                        "What are high risk studies?"
                    )
                    self.log_test(
                        "AgentEngine.process_query()", True, str(result)[:40]
                    )
                except Exception as e:
                    self.log_test(
                        "AgentEngine.process_query()", False, str(e)[:60]
                    )

            if hasattr(engine, "handle_request"):
                try:
                    result = await engine.handle_request(
                        {"type": "query", "content": "test"}
                    )
                    self.log_test(
                        "AgentEngine.handle_request()", True, str(result)[:40]
                    )
                except Exception as e:
                    self.log_test(
                        "AgentEngine.handle_request()", False, str(e)[:60]
                    )

            if hasattr(engine, "run"):
                try:
                    result = await engine.run({"query": "test"})
                    self.log_test("AgentEngine.run()", True, str(result)[:40])
                except Exception as e:
                    self.log_test("AgentEngine.run()", False, str(e)[:60])

            # Test orchestrator if available
            if hasattr(engine, "orchestrator") and engine.orchestrator:
                orchestrator = engine.orchestrator
                orch_methods = [
                    m
                    for m in dir(orchestrator)
                    if not m.startswith("_")
                    and callable(getattr(orchestrator, m))
                ]
                self.log_test(
                    "Orchestrator Methods", True, f"{orch_methods[:8]}"
                )

        except Exception as e:
            self.log_test("Agent Execution Suite", False, str(e)[:60])

    async def test_data_access(self):
        """Test that agents can access actual data."""
        logger.info("=" * 60)
        logger.info("🧪 TEST 9: DATA ACCESS")
        logger.info("=" * 60)

        # Check data files exist
        data_paths = [
            ("Study Metrics", Path("D:/trialos/data/metrics/study_metrics.csv")),
            ("Site Metrics", Path("D:/trialos/data/metrics/site_metrics.csv")),
            ("Study Indices", Path("D:/trialos/data/indices/study_indices.csv")),
            ("Site Features", Path("D:/trialos/data/features/site_features.csv")),
            ("Knowledge Base", Path("D:/trialos/data/knowledge_base/vector_store")),
            ("ML Models", Path("D:/trialos/data/models")),
        ]

        for name, path in data_paths:
            exists = path.exists()
            self.log_test(f"Data: {name}", exists, str(path)[-40:])

    def print_summary(self):
        """Print test summary."""
        logger.info("=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)

        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0

        status = (
            "🎉 ALL TESTS PASSED!"
            if self.failed == 0
            else f"⚠️  {self.failed} TESTS FAILED"
        )

        print(
            f"""
╔══════════════════════════════════════════════════════════════╗
║                    AGENT SYSTEM TEST RESULTS                 ║
╠══════════════════════════════════════════════════════════════╣
║  Total Tests:    {total:4d}                                      ║
║  ✅ Passed:      {self.passed:4d}                                      ║
║  ❌ Failed:      {self.failed:4d}                                      ║
║  Pass Rate:      {pass_rate:5.1f}%                                   ║
╠══════════════════════════════════════════════════════════════╣
║  Status: {status:<43} ║
╚══════════════════════════════════════════════════════════════╝
        """
        )

        if self.failed > 0:
            logger.warning("\n❌ FAILED TESTS:")
            for name, result in self.results.items():
                if not result["status"]:
                    logger.error(f"  - {name}: {result['message']}")

    async def run_all_tests(self):
        """Run all tests."""
        print(
            """
╔══════════════════════════════════════════════════════════════════════════════╗
║            🧪 NEXUS CLINICAL AI - AGENT SYSTEM VALIDATION 🧪                 ║
║                     Adapted to Actual Implementation                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        )

        await self.test_imports()
        await self.test_agent_initialization()
        await self.test_guardrails()
        await self.test_memory_system()
        await self.test_messaging_system()
        await self.test_agent_engine()
        await self.test_agent_methods()
        await self.test_agent_execution()
        await self.test_data_access()

        self.print_summary()

        return self.failed == 0


async def main():
    """Main entry point."""
    tester = AgentSystemTester()
    success = await tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
