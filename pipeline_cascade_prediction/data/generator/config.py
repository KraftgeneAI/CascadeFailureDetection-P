"""
config.py (Pipeline Digital Twin)
=================================
Central configuration for the pipeline cascade failure simulator and ML model.

All hardcoded numeric constants live here. Import with:
    from pipeline_cascade_prediction.data.generator.config import Settings
"""

# ---------------------------------------------------------------------------
# Pipeline System Base
# ---------------------------------------------------------------------------
class PipelineSystemConfig:
    REFERENCE_PRESSURE_PSI = 1000.0   # Base pressure for the main injection slack node
    BASE_FLOW_BBL_HR       = 1000.0   # Base flow rate unit for normalization


# ---------------------------------------------------------------------------
# Transient & Surge Dynamics
# ---------------------------------------------------------------------------
class SurgeConfig:
    BASE_SURGE_METRIC   = 1.0         # 1.0 = stable steady-state flow
    SURGE_DAMPING_MIN   = 0.05
    SURGE_DAMPING_MAX   = 0.15
    SURGE_TAU_MINUTES   = 0.5         # First-order lag for shockwave propagation
    SURGE_MAX_LIMIT     = 2.0         # Hard cap before catastrophic surge failure


# ---------------------------------------------------------------------------
# Thermal Dynamics
# ---------------------------------------------------------------------------
class ThermalConfig:
    AMBIENT_TEMP_C      = 15.0        # Default ambient ground/air temperature (°C)
    DT_MINUTES          = 2.0         # Timestep for temperature update (minutes)
    TEMP_NOISE_STD      = 0.2         # Gaussian noise std dev (°C)
    TEMP_MAX_C          = 150.0       # Equipment temperature upper limit (°C)
    DIURNAL_AMPLITUDE_C = 8.0         # ±°C diurnal swing around base ambient


# ---------------------------------------------------------------------------
# Topology Generation
# ---------------------------------------------------------------------------
class TopologyConfig:
    """Topology derived from the Seaway crude oil pipeline system.

    Reference
    ---------
    Zlotnik et al., "Optimal Economic Operation of Liquid Petroleum Products
    Pipeline Systems", arXiv:2012.11755. Their published test case is a 23-node
    tree (13 pipes, 9 pumps, 3 producers at N1/N9/N18, 2 consumers at N15/N23).
    That is too small and too structurally trivial to train a GNN on - a tree
    has no redundancy, so every internal node is an articulation point and the
    cascade extent is fully determined by graph position.

    We therefore follow the same approach the paper itself used ("synthesized
    based on physical and economic aspects on the Seaway Pipeline System using
    openly available information ... added fictitious supply and consumption
    points to create a richer variety of possible solutions"): keep the real
    route length, diameter, throughput and pump-station count/spacing, and
    subdivide the trunk plus add delivery branches to reach a useful node count.

    Real Seaway figures used below are from §5.2 of that paper.
    """
    DEFAULT_NUM_NODES       = 118       # Default pipeline network size

    # --- real Seaway system parameters -------------------------------------
    ROUTE_LENGTH_KM         = 968.0     # 601 miles, Cushing OK -> Freeport TX
    PIPE_DIAMETER_M         = 0.76      # 30 inch
    THROUGHPUT_M3_H         = 6300.0    # 950,000 bbl/day
    NUM_PUMP_STATIONS       = 9         # P1..P9, in series along the trunk
    #: Pump station spacing follows from 968 km / 9 stations ~= 108 km.

    # The Seaway system is twinned: the original line plus the "Seaway Twin"
    # (loop) line, 512 miles / 824 km of parallel 30-inch pipe built along the
    # same route, which more than doubled system capacity to 850,000 bbl/day.
    # Modelling the corridor as two cross-connected parallel lines rather than
    # a single chain is therefore faithful to the real system - and it is what
    # keeps the graph from degenerating into a path where ~90% of nodes are
    # single points of failure.
    TWIN_ROUTE_KM           = 824.0     # 512 miles, Cushing -> Jones Creek
    TWIN_FRACTION           = 0.85      # share of the route that is twinned
    #: Interconnects between the two lines. Pump stations always get one; the
    #: rest are spaced along the twinned corridor, as real twinned systems are
    #: cross-connected between stations for operational flexibility.
    CROSSOVER_SPACING_KM    = 55.0

    # --- how the real system is scaled up to DEFAULT_NUM_NODES -------------
    TRUNK_FRACTION          = 0.72      # share of nodes forming the twinned corridor
    #: Delivery groups along the route, as (fraction of route, number of
    #: terminals). Mirrors the real delivery points: ECHO/Texas docks and
    #: Freeport mid-route, Jones Creek at the end, and the Beaumont /
    #: Port Arthur group where Seaway meets three terminals.
    DELIVERY_GROUPS         = [(0.46, 2), (0.63, 2), (0.80, 3), (0.93, 3), (1.00, 2)]
    BRANCH_LENGTH_KM_MIN    = 4.0       # lateral spur length
    BRANCH_LENGTH_KM_MAX    = 22.0
    LOOP_PROBABILITY        = 0.55      # parallel redundancy loop at a pump station
    ELEVATION_MIN_M         = -20.0     # Cushing plateau -> Gulf coast sea level
    ELEVATION_MAX_M         = 320.0


# ---------------------------------------------------------------------------
# Time-Series Simulation
# ---------------------------------------------------------------------------
class SimulationConfig:
    DEFAULT_SEQUENCE_LENGTH = 30        # Timesteps per scenario
    RAMP_FRACTION_MIN       = 0.65      # Minimum fraction of sequence used for stress ramp
    RAMP_FRACTION_MAX       = 0.85      # Maximum fraction of sequence used for stress ramp
    COLLAPSE_FAILURE_RATIO  = 0.9       # Fraction of failed nodes = total system collapse
    CASCADE_MAX_SPREAD_FRACTION = 0.30  # Max additional failures per cascade wave
    AMBIENT_BASE_MIN_C      = 10.0
    AMBIENT_BASE_MAX_C      = 25.0      


# ---------------------------------------------------------------------------
# Scenario Orchestration
# ---------------------------------------------------------------------------
class ScenarioConfig:
    DEFAULT_NUM_NODES       = TopologyConfig.DEFAULT_NUM_NODES
    DEFAULT_SEQUENCE_LENGTH = SimulationConfig.DEFAULT_SEQUENCE_LENGTH
    DEFAULT_NUM_NORMAL      = 100
    DEFAULT_NUM_CASCADE     = 80
    DEFAULT_NUM_STRESSED    = 20
    DEFAULT_BATCH_SIZE      = 1
    DEFAULT_SEED            = 42
    MAX_RETRIES             = 10

    CASCADE_STRESS_MIN      = 1.29
    CASCADE_STRESS_MAX      = 2.00
    STRESSED_STRESS_MIN     = 0.55
    STRESSED_STRESS_MAX     = 1.29
    NORMAL_STRESS_MIN       = 0.00
    NORMAL_STRESS_MAX       = 0.55


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DatasetConfig:
    DEFAULT_TOPOLOGY_FILE   = "data/pipeline_topology.pkl"
    TRAIN_RATIO             = 0.70
    VAL_RATIO               = 0.15
    TEST_RATIO              = 0.15
    RATIO_TOLERANCE         = 1e-6      


# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
class TrainingConfig:
    LEARNING_RATE           = 0.0001    
    GRAD_CLIP               = 20.0      
    TRAINER_MAX_GRAD_NORM   = 5.0       
    EPOCHS                  = 100       
    BATCH_SIZE              = 8         
    PATIENCE                = 10        
    WEIGHT_DECAY            = 1e-3      
    SCHEDULER_PATIENCE      = 5         
    CASCADE_THRESHOLD       = 0.25      
    NODE_THRESHOLD          = 0.25      
    FBETA                   = 1.0       


# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
class ModelConfig:
    EMBEDDING_DIM       = 128           
    HIDDEN_DIM          = 128           
    NUM_GNN_LAYERS      = 3             
    HEADS               = 4             
    DROPOUT             = 0.3           
    DROPOUT_TRAIN       = 0.5           
    HEAD_DROPOUT_HIGH   = 0.4           
    HEAD_DROPOUT_LOW    = 0.3           
    LSTM_NUM_LAYERS     = 3             
    LSTM_DROPOUT        = 0.3           
    EDGE_FEATURES       = 3             # Pipe resistance, Flow limits, Pipe flows
    RISK_DIM            = 7             
    GAT_DROPOUT         = 0.1           
    LEAKY_RELU_SLOPE    = 0.2           


# ---------------------------------------------------------------------------
# Physics Supervision
# ---------------------------------------------------------------------------
class PhysicsPredConfig:
    LAMBDA_PRESSURE = 0.5   # MSE loss for pressure predictions
    LAMBDA_TEMP     = 0.5   # MSE loss for temperature predictions
    LAMBDA_FLOW     = 0.3   # MSE loss for fluid flow predictions


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------
class LossConfig:
    LAMBDA_PREDICTION   = 30.0
    LAMBDA_RISK         = 0.1
    LAMBDA_TIMING       = 2.0
    LAMBDA_PARENT       = 0.3   
    PARENT_NON_TRIGGER_WEIGHT = 5.0  

    FOCAL_ALPHA         = 0.85          
    FOCAL_GAMMA         = 2.0
    FOCAL_ALPHA_TRAIN   = 0.75          
    FOCAL_ALPHA_FALLBACK = 0.65         


# ---------------------------------------------------------------------------
# Embedding Networks
# ---------------------------------------------------------------------------
class EmbeddingConfig:
    DROPOUT_FC              = 0.3       

    NODE_FEATURE_DIM        = 124       # Full per-node feature vector width 
    NODE_MLP_HIDDEN_1       = 256       
    NODE_MLP_HIDDEN_2       = 128       


# ---------------------------------------------------------------------------
# Single entry-point
# ---------------------------------------------------------------------------
class Settings:
    """Aggregates all config sub-classes under one name."""
    PipelineSystem = PipelineSystemConfig
    Surge         = SurgeConfig
    Thermal       = ThermalConfig
    Topology      = TopologyConfig
    Simulation    = SimulationConfig
    Scenario      = ScenarioConfig
    Dataset       = DatasetConfig
    Training      = TrainingConfig
    Model         = ModelConfig
    Loss          = LossConfig
    Embedding     = EmbeddingConfig
    PhysicsPred   = PhysicsPredConfig