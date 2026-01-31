import torch
import copy
import os
from typing import Dict, Any, Optional, Tuple
import logging





logger = logging.getLogger(__name__)







class ModelStateManager:

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.saved_states = {}

    def save_state(self, state_name: str = "baseline") -> Dict[str, torch.Tensor]:
        logger.info(f"Saving model state: {state_name}")

        state_dict = {}
        for name, param in self.model.state_dict().items():
            if param is not None:

                state_dict[name] = param.clone().detach()
            else:
                state_dict[name] = None

        self.saved_states[state_name] = state_dict


        total_params = sum(p.numel() for p in state_dict.values() if p is not None)
        logger.info(f"Saved state '{state_name}' with {len(state_dict)} parameters ({total_params:,} elements)")

        return state_dict



    def restore_state(self, state_name: str = "baseline") -> None:
        if state_name not in self.saved_states:
            raise ValueError(f"State '{state_name}' not found. Available states: {list(self.saved_states.keys())}")

        logger.info(f"Restoring model state: {state_name}")


        saved_state = self.saved_states[state_name]


        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in saved_state and saved_state[name] is not None:
                    param.copy_(saved_state[name])


        for name, buffer in self.model.named_buffers():
            if name in saved_state and saved_state[name] is not None:
                buffer.copy_(saved_state[name])

        logger.info(f"Successfully restored state '{state_name}'")



    def compare_states(self, state_name1: str, state_name2: str) -> Dict[str, Any]:
        if state_name1 not in self.saved_states:
            raise ValueError(f"State '{state_name1}' not found")
        if state_name2 not in self.saved_states:
            raise ValueError(f"State '{state_name2}' not found")

        state1 = self.saved_states[state_name1]
        state2 = self.saved_states[state_name2]


        differences = {}
        total_params = 0
        changed_params = 0
        total_diff = 0.0
        max_diff = 0.0
        changed_layers = []

        for name in state1.keys():
            if name not in state2:
                continue

            param1 = state1[name]
            param2 = state2[name]

            if param1 is None or param2 is None:
                continue

            diff = torch.abs(param1 - param2)
            num_changed = (diff > 0).sum().item()

            if num_changed > 0:
                changed_layers.append(name)
                changed_params += num_changed
                total_diff += diff.sum().item()
                max_diff = max(max_diff, diff.max().item())

                differences[name] = {
                    'num_changed': num_changed,
                    'total_elements': param1.numel(),
                    'percent_changed': 100.0 * num_changed / param1.numel(),
                    'mean_diff': diff.sum().item() / num_changed if num_changed > 0 else 0.0,
                    'max_diff': diff.max().item()
                }

            total_params += param1.numel()



        comparison_stats = {
            'total_parameters': total_params,
            'changed_parameters': changed_params,
            'percent_changed': 100.0 * changed_params / total_params if total_params > 0 else 0.0,
            'total_diff': total_diff,
            'max_diff': max_diff,
            'num_changed_layers': len(changed_layers),
            'changed_layers': changed_layers,
            'layer_details': differences
        }

        logger.info(f"Comparison {state_name1} vs {state_name2}:")
        logger.info(f"  - Changed parameters: {changed_params:,} / {total_params:,} ({comparison_stats['percent_changed']:.4f}%)")
        logger.info(f"  - Changed layers: {len(changed_layers)}")
        logger.info(f"  - Max difference: {max_diff:.6e}")

        return comparison_stats




    def save_to_disk(self, filepath: str, state_name: str = "baseline") -> None:
        if state_name not in self.saved_states:
            raise ValueError(f"State '{state_name}' not found")



        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.saved_states[state_name], filepath)
        logger.info(f"Saved state '{state_name}' to {filepath}")





    def load_from_disk(self, filepath: str, state_name: str = "loaded") -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"State file not found: {filepath}")

        self.saved_states[state_name] = torch.load(filepath)
        logger.info(f"Loaded state '{state_name}' from {filepath}")




    def clear_state(self, state_name: Optional[str] = None) -> None:
        if state_name is None:
            self.saved_states.clear()
            logger.info("Cleared all saved states")
        elif state_name in self.saved_states:
            del self.saved_states[state_name]
            logger.info(f"Cleared state '{state_name}'")
        else:
            logger.warning(f"State '{state_name}' not found, nothing to clear")





def save_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    state_dict = {}
    for name, param in model.state_dict().items():
        if param is not None:
            state_dict[name] = param.clone().detach()
        else:
            state_dict[name] = None
    return state_dict


def restore_model_state(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state_dict and state_dict[name] is not None:
                param.copy_(state_dict[name])

        for name, buffer in model.named_buffers():
            if name in state_dict and state_dict[name] is not None:
                buffer.copy_(state_dict[name])


def compare_model_states(state1: Dict[str, torch.Tensor],
                         state2: Dict[str, torch.Tensor]) -> Tuple[int, int, float]:
    total_params = 0
    changed_params = 0

    for name in state1.keys():
        if name not in state2:
            continue

        param1 = state1[name]
        param2 = state2[name]

        if param1 is None or param2 is None:
            continue

        diff = torch.abs(param1 - param2)
        num_changed = (diff > 0).sum().item()
        changed_params += num_changed
        total_params += param1.numel()




    percent_changed = 100.0 * changed_params / total_params if total_params > 0 else 0.0

    return changed_params, total_params, percent_changed
