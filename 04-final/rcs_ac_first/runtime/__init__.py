"""Runtime contracts for RCS-AC-first."""

from rcs_ac_first.runtime.action import (
    STAGE4_ACTION_V2,
    action_contract_spec,
    action_dim_for_contract,
    clamp_action_tensor,
    decode_stage4_action_v2,
    validate_action_tensor,
)
from rcs_ac_first.runtime.config import (
    Phase1CoreConfig,
    RCSACFirstFusionConfig,
    RCSACFirstLossConfig,
    RCSACFirstModelConfig,
    RCSACFirstTrainConfig,
    load_phase1_config,
)
from rcs_ac_first.runtime.auxiliary import (
    AuxiliaryObservationSpec,
    build_auxiliary_observation_spec,
    flatten_auxiliary_observation_dict,
    split_flat_auxiliary_observation,
)
from rcs_ac_first.runtime.observation import (
    DEFAULT_EGO_DIM,
    DEFAULT_LIDAR_SHAPE,
    DEFAULT_RGB_SHAPE,
    DEFAULT_ROUTE_DIM,
    ObservationSliceSpec,
    PolicyObservationSpec,
    build_policy_observation_spec,
    flatten_observation_dict,
    split_flat_observation,
)

__all__ = [
    "DEFAULT_EGO_DIM",
    "DEFAULT_LIDAR_SHAPE",
    "DEFAULT_RGB_SHAPE",
    "DEFAULT_ROUTE_DIM",
    "ObservationSliceSpec",
    "AuxiliaryObservationSpec",
    "PolicyObservationSpec",
    "Phase1CoreConfig",
    "RCSACFirstFusionConfig",
    "RCSACFirstLossConfig",
    "RCSACFirstModelConfig",
    "RCSACFirstTrainConfig",
    "STAGE4_ACTION_V2",
    "action_contract_spec",
    "action_dim_for_contract",
    "build_policy_observation_spec",
    "build_auxiliary_observation_spec",
    "clamp_action_tensor",
    "decode_stage4_action_v2",
    "flatten_observation_dict",
    "flatten_auxiliary_observation_dict",
    "load_phase1_config",
    "split_flat_observation",
    "split_flat_auxiliary_observation",
    "validate_action_tensor",
]
