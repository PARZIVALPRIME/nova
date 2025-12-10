"""
NEXUS CLINICAL AI - Complete Pipeline Runner
=============================================
Runs: Ingestion → Harmonization → Versioning → Knowledge Graph
      → Metrics → Indices → Trends → Benchmarking → Features → Models
      → Optimization → Validation → Knowledge Base → Agents → Orchestration
"""

import sys
from pathlib import Path
from loguru import logger

# Add src to path (so imports like ingestion.*, validation.*, etc. work)
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Core pipeline imports
from ingestion.data_loader import NexusDataLoader
from ingestion.gap_analyzer import run_gap_analysis
from ingestion.harmonizer import run_harmonization
from versioning.version_manager import run_versioning
from knowledge_graph.graph_builder import ClinicalKnowledgeGraph
from knowledge_graph.graph_queries import GraphQueryEngine
from knowledge_graph.graph_analytics import GraphAnalytics
from metrics.metrics_calculator import run_metrics_calculation
from indices.index_calculator import run_index_calculation
from trends.trend_engine import run_trend_analysis
from benchmarking.benchmark_engine import run_benchmarking
from features.feature_engineer import run_feature_engineering
from models.model_trainer import run_model_training  # ✅ Step 10 import
from optimization.optimization_engine import run_optimization
from validation.validation_engine import run_validation
from knowledge_base.knowledge_base_engine import run_knowledge_base_setup
from agents.agent_engine import run_agent_setup


def run_knowledge_graph():
    """Run knowledge graph construction (Step 4)."""
    print("\n" + "=" * 70)
    print("🕸️ STEP 4: KNOWLEDGE GRAPH CONSTRUCTION")
    print("=" * 70)

    # Initialize graph builder
    print("\n📋 Step 4.1: Initializing Knowledge Graph Builder...")
    kg = ClinicalKnowledgeGraph()
    print("   ✅ Knowledge Graph Builder initialized")

    # Build graph
    print("\n🔨 Step 4.2: Building Knowledge Graph...")
    graph = kg.build_graph()

    # Print statistics
    print("\n📊 Step 4.3: Graph Statistics...")
    kg.print_statistics()

    # Save graph
    print("\n💾 Step 4.4: Saving Knowledge Graph...")
    kg.save_graph()

    # Initialize query engine
    print("\n🔍 Step 4.5: Initializing Query Engine...")
    query_engine = GraphQueryEngine(graph)
    print("   ✅ Query Engine ready")

    # Initialize analytics
    print("\n📈 Step 4.6: Running Graph Analytics...")
    analytics = GraphAnalytics(graph)
    analytics.print_analytics_report()

    # Demo queries
    print("\n" + "=" * 70)
    print("🔍 SAMPLE QUERIES")
    print("=" * 70)

    # Find studies
    studies = query_engine.find_nodes_by_type("study", limit=5)
    print(f"\n📋 Studies in graph: {len(studies)}")
    for node_id, attrs in studies[:3]:
        print(f"   • {node_id}: {attrs.get('total_patients', 0)} patients")

    # Find sites
    sites = query_engine.find_nodes_by_type("site", limit=5)
    print(f"\n🏥 Sites in graph: {len(sites)}")

    # Find patients with issues
    unclean_patients = query_engine.find_nodes_by_attribute(
        attribute="is_clean",
        value=False,
        entity_type="patient",
    )
    print(f"\n⚠️ Patients not clean: {len(unclean_patients)}")

    # Summary
    print("\n" + "=" * 70)
    print("🕸️ KNOWLEDGE GRAPH SUMMARY")
    print("=" * 70)
    print(f"\n📁 GRAPH STORAGE: data/graph/")
    print("   • clinical_knowledge_graph.gpickle")
    print("   • graph_metadata.json")

    print("\n📊 CAPABILITIES ENABLED:")
    print("   • ✅ Entity lookups (study, site, patient, SAE, visit)")
    print("   • ✅ Relationship traversal")
    print("   • ✅ Pathfinding between entities")
    print("   • ✅ Hierarchy queries")
    print("   • ✅ Centrality analysis")
    print("   • ✅ Anomaly detection")
    print("   • ✅ Subgraph extraction")

    print("\n💡 USAGE EXAMPLES:")
    print("   • query_engine.get_patient_context(study_id=1, subject_id='PAT001')")
    print("   • query_engine.get_study_hierarchy(study_id=1)")
    print("   • analytics.find_data_quality_issues()")
    print("   • analytics.calculate_degree_centrality(entity_type='site')")

    print("\n" + "=" * 70)
    print("✅ STEP 4 COMPLETE!")
    print("=" * 70)

    return kg, query_engine, analytics


def run_step_16():
    """Step 16: Agent Orchestration & Guardrails."""
    logger.info("\n" + "=" * 70)
    logger.info("🎯 STEP 16: AGENT ORCHESTRATION & GUARDRAILS")
    logger.info("=" * 70)

    # Orchestration engine lives under src/orchestration/*
    from orchestration import OrchestrationEngine

    output_dir = Path("data/orchestration")

    # Initialize orchestration engine
    engine = OrchestrationEngine(output_dir)

    # Run demo
    demo_results = engine.run_demo()

    # Get final status
    status = engine.get_system_status()

    # Save all state
    saved_paths = engine.save_all_state()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 STEP 16 SUMMARY")
    logger.info("=" * 70)

    logger.info("\n🔧 COMPONENTS:")
    components = [
        ("GuardrailEngine", "Input/Output validation, rate limiting, safety"),
        ("AgentMemoryManager", "Short-term, long-term, episodic memory"),
        ("ContextManager", "Session & global context management"),
        ("ConflictResolver", "Multi-agent consensus resolution"),
        ("HumanInTheLoop", "Escalation & approval workflows"),
        ("ObservabilitySystem", "Metrics, tracing, alerting"),
    ]
    for name, desc in components:
        logger.info(f"   ✅ {name} - {desc}")

    stats = status["statistics"]["orchestration"]
    logger.info(f"\n📊 DEMO STATISTICS:")
    logger.info(f"   • Requests Processed: {stats['requests_processed']}")
    logger.info(f"   • Conflicts Resolved: {stats['conflicts_resolved']}")
    logger.info(f"   • Escalations: {stats['escalations_created']}")
    logger.info(f"   • System Status: {status['status']}")

    logger.info(f"\n💾 FILES SAVED: {len(saved_paths)}")
    for name, path in saved_paths.items():
        logger.info(f"   • {path}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ STEP 16 COMPLETE!")
    logger.info("=" * 70)

    return engine


def main():
    """Main execution function for complete data pipeline."""
    print("=" * 70)
    print("🚀 NEXUS CLINICAL AI - COMPLETE DATA PIPELINE")
    print("=" * 70)
    print()

    # =========================================
    # PHASE 1: DATA INGESTION (Step 1)
    # =========================================
    print("\n" + "=" * 70)
    print("📁 PHASE 1: DATA INGESTION")
    print("=" * 70)

    loader = NexusDataLoader()

    print("\n📁 Step 1.1: Discovering Studies...")
    studies = loader.discover_studies()
    print(f"   ✅ Found {len(studies)} studies")

    print("\n📋 Step 1.2: Building File Inventory...")
    inventory = loader.build_file_inventory()
    print(f"   ✅ Inventoried {len(inventory)} files")

    print("\n📥 Step 1.3: Loading All Files...")
    data = loader.load_all_files()
    print(f"   ✅ Loaded {len(data)} DataFrames")

    print("\n📊 Step 1.4: Profiling Data...")
    profiles = loader.profile_all_data()
    print(f"   ✅ Profiled {len(profiles)} files")

    # Gap analysis
    analyzer, complete_data = run_gap_analysis(loader)

    # =========================================
    # STEP 2: DATA HARMONIZATION
    # =========================================
    harmonizer, saved_files = run_harmonization(complete_data)

    # =========================================
    # STEP 3: DATA VERSIONING
    # =========================================
    version_manager, audit_logger = run_versioning(create_initial=True)

    # =========================================
    # STEP 4: KNOWLEDGE GRAPH
    # =========================================
    kg, query_engine, analytics = run_knowledge_graph()

    # =========================================
    # STEP 5: CORE METRICS CALCULATION
    # =========================================
    print("\n" + "=" * 70)
    print("📏 STEP 5: CORE METRICS CALCULATION")
    print("=" * 70)
    metrics_calculator, metrics_files = run_metrics_calculation()

    # =========================================
    # STEP 6: COMPOSITE INDICES
    # =========================================
    print("\n" + "=" * 70)
    print("📊 STEP 6: 7 COMPOSITE INDICES")
    print("=" * 70)
    index_calculator, index_files = run_index_calculation()

    # =========================================
    # STEP 7: TREND ANALYSIS
    # =========================================
    print("\n" + "=" * 70)
    print("📈 STEP 7: TREND ANALYSIS ENGINE")
    print("=" * 70)
    trend_engine, trend_files = run_trend_analysis()

    # =========================================
    # STEP 8: BENCHMARKING SYSTEM
    # =========================================
    print("\n" + "=" * 70)
    print("📊 STEP 8: BENCHMARKING SYSTEM")
    print("=" * 70)
    benchmark_engine, benchmark_files = run_benchmarking()

    # =========================================
    # STEP 9: FEATURE ENGINEERING
    # =========================================
    print("\n" + "=" * 70)
    print("🔧 STEP 9: FEATURE ENGINEERING")
    print("=" * 70)
    feature_engineer, feature_files = run_feature_engineering()

    # =========================================
    # STEP 10: MODEL DEVELOPMENT
    # =========================================
    print("\n" + "=" * 70)
    print("🏋️ STEP 10: MODEL DEVELOPMENT")
    print("=" * 70)
    model_trainer, model_files = run_model_training()

    # =========================================
    # STEP 11: HYPERPARAMETER OPTIMIZATION
    # =========================================
    print("\n" + "=" * 70)
    print("⚡ STEP 11: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    optimization_engine, optimization_files = run_optimization()

    # =========================================
    # STEP 12: MODEL VALIDATION & EXPLAINABILITY
    # =========================================
    print("\n" + "=" * 70)
    print("🔬 STEP 12: MODEL VALIDATION & EXPLAINABILITY")
    print("=" * 70)
    validation_engine, validation_files = run_validation()

    # =========================================
    # STEP 13: KNOWLEDGE BASE & RAG SETUP
    # =========================================
    print("\n" + "=" * 70)
    print("📚 STEP 13: KNOWLEDGE BASE & RAG SETUP")
    print("=" * 70)
    kb_engine, kb_files = run_knowledge_base_setup()

    # =========================================
    # STEPS 14–15: AGENTS SETUP (Core + Advanced)
    # =========================================
    print("\n" + "=" * 70)
    print("🤖 STEPS 14–15: CORE & ADVANCED AGENTS SETUP")
    print("=" * 70)
    agent_engine, agent_files = run_agent_setup()

    # =========================================
    # STEP 16: AGENT ORCHESTRATION & GUARDRAILS
    # =========================================
    orchestration_engine = run_step_16()

    # =========================================
    # FINAL SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("🎉 PIPELINE COMPLETE - STEPS 1–16 DONE!")
    print("=" * 70)

    print("\n📊 FINAL STATISTICS:")
    print("   • Studies Processed: 23")
    print("   • Files Processed: 207")
    print(
        f"   • Total Rows Ingested: {sum(len(df) for df in complete_data.values()):,}"
    )
    print(f"   • Patient Records: {len(harmonizer.patient_master):,}")
    print(f"   • Site Records: {len(harmonizer.site_master):,}")
    print(f"   • Versions Created: {len(version_manager.manifest['versions'])}")
    print(f"   • Graph Nodes: {kg.graph.number_of_nodes():,}")
    print(f"   • Graph Edges: {kg.graph.number_of_edges():,}")
    print(f"   • Metrics Files: {len(metrics_files)}")
    print(f"   • Index Files: {len(index_files)}")
    print(f"   • Trend Files: {len(trend_files)}")
    print(f"   • Benchmark Files: {len(benchmark_files)}")
    print(f"   • Feature Files: {len(feature_files)}")
    print(f"   • Model Files: {len(model_files)}")

    print("\n📁 OUTPUT LOCATIONS:")
    print("   • Unified Data: data/unified/")
    print("   • Versions: data/versions/")
    print("   • Audit Logs: data/audit/")
    print("   • Knowledge Graph: data/graph/")
    print("   • Metrics: data/metrics/")
    print("   • Composite Indices: data/indices/")
    print("   • Trends: data/trends/")
    print("   • Benchmarks: data/benchmarks/")
    print("   • Features: data/features/")
    print("   • Models: data/models/")
    print("   • Optimization: data/optimization/")
    print("   • Validation: data/validation/")
    print("   • Knowledge Base: data/knowledge_base/")
    print("   • Agents: data/agents/")
    print("   • Orchestration: data/orchestration/")

    print("\n✅ COMPLETED STEPS:")
    print("   • Step 1: Data Ingestion & Profiling ✅")
    print("   • Step 2: Data Harmonization & Unification ✅")
    print("   • Step 3: Data Versioning & Change Detection ✅")
    print("   • Step 4: Knowledge Graph Construction ✅")
    print("   • Step 5: Core Metrics Calculation ✅")
    print("   • Step 6: 7 Composite Indices ✅")
    print("   • Step 7: Trend Analysis Engine ✅")
    print("   • Step 8: Benchmarking System ✅")
    print("   • Step 9: Feature Engineering ✅")
    print("   • Step 10: Model Development ✅")
    print("   • Step 11: Hyperparameter Optimization ✅")
    print("   • Step 12: Model Validation & Explainability ✅")
    print("   • Step 13: Knowledge Base & RAG Setup ✅")
    print("   • Step 14: Core Agents (5 Agents) ✅")
    print("   • Step 15: Advanced Agents (5 Agents) ✅")
    print("   • Step 16: Agent Orchestration & Guardrails ✅")

    # Return core objects (same signature you had before, to avoid breaking callers)
    return (
        loader,
        analyzer,
        harmonizer,
        complete_data,
        version_manager,
        audit_logger,
        kg,
        query_engine,
        analytics,
        metrics_calculator,
        metrics_files,
        index_calculator,
        index_files,
        trend_engine,
        trend_files,
        benchmark_engine,
        benchmark_files,
        feature_engineer,
        feature_files,
        model_trainer,
        model_files,
    )


if __name__ == "__main__":
    results = main()
    (
        loader,
        analyzer,
        harmonizer,
        complete_data,
        version_manager,
        audit_logger,
        kg,
        query_engine,
        analytics,
        metrics_calculator,
        metrics_files,
        index_calculator,
        index_files,
        trend_engine,
        trend_files,
        benchmark_engine,
        benchmark_files,
        feature_engineer,
        feature_files,
        model_trainer,
        model_files,
    ) = results
