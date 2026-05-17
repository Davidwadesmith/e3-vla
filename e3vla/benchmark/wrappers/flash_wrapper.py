"""FLASH-style speculative VLA wrapper.

Thin wrapper (~100 lines) around dexmal/realtime-vla-flash.
Does NOT start a server — directly imports core Python classes.
"""

from typing import Optional
import torch

from e3vla.schema import Observation, ActionCommand
from e3vla.utils import compute_ee_pose


class FLASHWrapper:
    """Wraps FLASH SpecPI0Pytorch behind the BenchmarkPolicy interface.

    Dependency: pip install realtime-vla-flash
    If not installed, wrapper raises RuntimeError with clear instructions.
    """

    def __init__(self, config: dict):
        self.chunk_len = config.get("chunk_len", 16)
        self.full_exec_len = config.get("full_exec_len", 2)

        try:
            from openpi.models_pytorch.spec_pi0_pytorch import SpecPI0Pytorch, SpecArgs

            spec_args = SpecArgs(
                t_list=config.get("t_list", (0.10, 0.05)),
                tau_radius=config.get("tau_radius", 0.3),
                dist_dims=config.get("dist_dims", 6),
                eval_h=config.get("eval_h", 12),
            )

            self._model = SpecPI0Pytorch(
                base_policy=config["base_policy_path"],
                draft_ckpt=config["draft_ckpt_path"],
                spec_args=spec_args,
            )
            self._available = True

        except ImportError:
            print("Warning: FLASH not installed. Install with: pip install realtime-vla-flash")
            self._model = None
            self._available = False

        # State
        self._draft_rounds_since_full = 0
        self._pending_full = True

    def reset(self, task_info=None) -> None:
        self._draft_rounds_since_full = 0
        self._pending_full = True

    def act(self, obs: Observation) -> ActionCommand:
        if not self._available:
            raise RuntimeError("FLASH not installed")

        x0, accepted_len, diagnostics = self._model.forward(
            obs.image, obs.instruction, obs.robot_state,
        )

        mode = "full_refresh" if self._pending_full else "speculative"
        self._pending_full = False

        return ActionCommand(
            actions=x0[:self.full_exec_len] if mode == "full_refresh" else x0[:accepted_len],
            execute_len=self.full_exec_len if mode == "full_refresh" else accepted_len,
            can_interrupt=True,
            mode=mode,
            prefix_length=accepted_len,
            diagnostics=diagnostics or {},
        )

    def get_diagnostics(self) -> dict:
        return {"method": "FLASH-style Drafter + Verifier"}

    @property
    def method_name(self) -> str:
        return "FLASH-style Drafter + Verifier"

    @property
    def method_type(self) -> str:
        return "drafter_verify"
