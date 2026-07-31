"""
config.py (Pipeline Digital Twin)
=================================
Central configuration for the pipeline cascade failure simulator and ML model.

All hardcoded numeric constants live here. Import with:
    from pipeline_twin.data.generator.config import Settings
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
    NUM_ZONES               = 4
    ZONE_CENTERS            = [(-50, -50), (50, -50), (-50, 50), (50, 50)]  # (x, y) km
    ZONE_SPREAD_STD         = 20.0      # Node position std dev within zone (km)
    INTRA_ZONE_CONN_MIN     = 2         # Min connections per node within zone
    INTRA_ZONE_CONN_MAX     = 5         # Max connections per node within zone
    TIE_LINES_MIN           = 2         # Min tie lines between adjacent zones
    TIE_LINES_MAX           = 4         # Max tie lines between adjacent zones
    DEFAULT_NUM_NODES       = 118       # Default pipeline network size


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

    # Stress level ranges by scenario type
    CASCADE_STRESS_MIN      = 1.30
    CASCADE_STRESS_MAX      = 2.00
    STRESSED_STRESS_MIN     = 0.55
    STRESSED_STRESS_MAX     = 1.30
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