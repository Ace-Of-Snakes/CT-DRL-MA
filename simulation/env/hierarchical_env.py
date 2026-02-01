# simulation/env/hierarchical_env.py
"""
Hierarchical environment step function for two-stage agent.

OPTIMIZED VERSION: Reduces encode_with_forecast calls.
- Original: 2N + 2 encodes per step_all_cranes (N = num cranes)
- Optimized: N + 1 encodes per step_all_cranes
- State passed between crane steps to avoid redundant initial encodes
- Time advances after every crane action (required for simulation correctness)
"""
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from numpy.typing import NDArray

from simulation.core.facilities.yard import PlacementResult
from simulation.core.enums import MoveType

from simulation.operations.hierarchical_moves import HierarchicalMoveGenerator
from simulation.rl.features.featurizers import (
    MoveableContainer, Destination, ParkingAction,
    SourceType, DestinationType
)
from simulation.rl.agents.hierarchical_dqn_agent import (
    HierarchicalDQNAgent, ActionPool, Stage1Selection, HierarchicalTransition
)


@dataclass
class HierarchicalStepResult:
    """Result of one hierarchical step."""
    next_state: NDArray[np.float32]
    reward: float
    done: bool
    info: Dict[str, Any]
    transition: Optional[HierarchicalTransition] = None
    was_valid_move: bool = False


class HierarchicalEnvWrapper:
    """
    Wrapper that adds hierarchical stepping to ContainerTerminalEnv.
    
    Optimized to minimize redundant state encoding calls.
    """
    
    def __init__(
        self,
        env,
        max_retries: int = 1,
        no_destination_penalty: float = -1.0
    ):
        self.env = env
        self.max_retries = max_retries
        self.no_destination_penalty = no_destination_penalty
        
        self.move_gen = HierarchicalMoveGenerator(
            yard=env.yard,
            rail=env.rail,
            parking=env.parking,
            proximity=5
        )
        
        self._train_heat_bays: Set[int] = set()
    
    def reset(self, *args, **kwargs):
        """Reset underlying environment."""
        result = self.env.reset(*args, **kwargs)
        self._update_train_heat()
        return result
    
    def _update_train_heat(self):
        """Update train heat bays from current trains."""
        self._train_heat_bays.clear()
        for train_id in self.env.trains:
            anchor = self.env.rail.get_anchor_bay(train_id)
            if anchor is not None:
                self._train_heat_bays.add(anchor)
    
    def _encode_state(self) -> NDArray[np.float32]:
        """Single point for state encoding."""
        return self.env.encoder.encode_with_forecast(
            self.env.trains,
            self.env.trucks,
            self.env.terminal_trucks,
            self.env.day_plan,
            self.env.current_time
        )
    
    def step_hierarchical(
        self,
        agent: HierarchicalDQNAgent,
        crane_id: int = 0,
        state_np: Optional[NDArray[np.float32]] = None
    ) -> HierarchicalStepResult:
        """
        Execute one hierarchical step for a single crane.
        
        Args:
            agent: The hierarchical DQN agent
            crane_id: Which crane is deciding
            state_np: Pre-computed state (avoids redundant encoding)
            
        Returns:
            HierarchicalStepResult
        """
        now = self.env.current_time
        info: Dict[str, Any] = {
            "crane_id": crane_id,
            "retries": 0,
            "executed": [],
        }
        
        # Clear agent's state cache for new decision
        agent.clear_state_cache()
        
        # Use provided state or encode (only if not provided)
        if state_np is None:
            state_np = self._encode_state()
        
        # Build action pool
        containers = self.move_gen.list_moveable_containers(
            self.env.trains,
            self.env.trucks,
            now
        )
        parkings = self.move_gen.list_parking_actions(self.env.trucks)
        pool = ActionPool(containers=containers, parkings=parkings)
        
        # Handle empty pool - advance time since we can't do anything
        if pool.is_empty():
            self._advance_time()
            next_state = self._encode_state()  # Re-encode for updated time features
            done = self._check_day_end()
            return HierarchicalStepResult(
                next_state=next_state,
                reward=0.0,
                done=done,
                info=info,
                transition=None,
                was_valid_move=False
            )
        
        # Retry loop for invalid container selections
        excluded_container_ids: Set[str] = set()
        total_penalty = 0.0
        
        for retry in range(self.max_retries):
            info["retries"] = retry
            
            # Filter out excluded containers
            filtered_containers = [
                c for c in containers
                if c.container_id not in excluded_container_ids
            ]
            filtered_pool = ActionPool(
                containers=filtered_containers,
                parkings=parkings
            )
            
            if filtered_pool.is_empty():
                break
            
            # Stage 1: Select action
            selection = agent.select_stage1(state_np, filtered_pool)
            
            if selection.index < 0:
                break
            
            # Handle parking action
            if selection.is_parking:
                result = self._execute_parking(
                    state_np, selection, filtered_pool, agent, info
                )
                result.reward += total_penalty
                return result
            
            # Handle container selection
            cont = selection.container
            
            # Stage 2: Get destinations for selected container
            destinations = self.move_gen.list_destinations_for_container(
                cont,
                self.env.trains,
                self.env.trucks,
                self._train_heat_bays
            )
            
            if not destinations:
                # No valid destinations - penalty and retry
                total_penalty += self.no_destination_penalty
                excluded_container_ids.add(cont.container_id)
                info.setdefault("retry_reasons", []).append(
                    f"no_dest:{cont.source_type.value}"
                )
                continue
            
            # Stage 2: Select destination
            dest_idx = agent.select_stage2(
                state_np,
                selection.container_feat,
                destinations,
                cont.bay,
                cont.tier
            )
            
            if dest_idx < 0 or dest_idx >= len(destinations):
                total_penalty += self.no_destination_penalty
                excluded_container_ids.add(cont.container_id)
                continue
            
            dest = destinations[dest_idx]
            
            # Execute the move
            result = self._execute_move(
                state_np, cont, selection, dest, dest_idx, destinations,
                filtered_pool, agent, info
            )
            result.reward += total_penalty
            return result
        
        # All retries exhausted - advance time
        self._advance_time()
        next_state = self._encode_state()  # Re-encode to get updated time features
        done = self._check_day_end()
        
        return HierarchicalStepResult(
            next_state=next_state,
            reward=total_penalty,
            done=done,
            info=info,
            transition=None,
            was_valid_move=False
        )
    
    def _execute_parking(
        self,
        state_np: NDArray[np.float32],
        selection: Stage1Selection,
        pool: ActionPool,
        agent: HierarchicalDQNAgent,
        info: Dict
    ) -> HierarchicalStepResult:
        """Execute parking action."""
        parking = selection.parking
        truck = self.env.trucks.get(parking.truck_id)
        
        if truck and self.env.parking:
            success = self.env.parking.allocate(truck, parking.spot)
        else:
            success = False
        
        # Advance time (parking takes time too)
        self._advance_time()
        
        if success:
            reward = 0.5
            info["executed"].append({
                "type": "PARKING",
                "truck_id": parking.truck_id,
                "spot": parking.spot
            })
        else:
            reward = -0.5
        
        # Always re-encode after time advances
        next_state = self._encode_state()
        done = self._check_day_end()
        
        # Build transition
        parking_feats = agent.park_featurizer.featurize_batch(pool.parkings).numpy()
        
        transition = HierarchicalTransition(
            state=state_np.copy(),
            container_feats=np.zeros((0, 16), dtype=np.float32),
            container_idx=-1,
            container_feat=np.zeros(16, dtype=np.float32),
            destination_feats=np.zeros((0, 12), dtype=np.float32),
            destination_idx=-1,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            was_parking=True,
            parking_feats=parking_feats,
            parking_idx=selection.index
        )
        
        return HierarchicalStepResult(
            next_state=next_state,
            reward=reward,
            done=done,
            info=info,
            transition=transition,
            was_valid_move=success
        )
    
    def _execute_move(
        self,
        state_np: NDArray[np.float32],
        cont: MoveableContainer,
        selection: Stage1Selection,
        dest: Destination,
        dest_idx: int,
        destinations: List[Destination],
        pool: ActionPool,
        agent: HierarchicalDQNAgent,
        info: Dict
    ) -> HierarchicalStepResult:
        """Execute container move."""
        move_type = self._determine_move_type(cont.source_type, dest.dest_type)
        args = self._build_move_args(cont, dest)
        
        # Execute through TLM
        success = self.env.tlm.execute(
            self._create_move(move_type, args),
            self.env.trains,
            self.env.trucks,
            self.env.terminal_trucks
        )
        
        # CRITICAL: Advance time after crane operation (crane takes time to move)
        self._advance_time()
        
        if success:
            reward = self.env.reward_engine.immediate_reward(
                move_type.value,
                distance_m=abs(dest.bay - cont.bay) * 2.0,
                time_s=30.0
            )
            info["executed"].append({
                "type": move_type.value,
                "container_id": cont.container_id,
                "source": cont.source_type.value,
                "dest": dest.dest_type.value
            })
            self._update_train_heat()
        else:
            reward = -0.5
        
        # Always re-encode after time advances (time features change)
        next_state = self._encode_state()
        done = self._check_day_end()
        
        # Build transition
        cont_feats = agent.cont_featurizer.featurize_batch(pool.containers).numpy()
        dest_feats = agent.dest_featurizer.featurize_batch(
            destinations, cont.bay, cont.tier
        ).numpy()
        
        transition = HierarchicalTransition(
            state=state_np.copy(),
            container_feats=cont_feats,
            container_idx=selection.index,
            container_feat=selection.container_feat.copy(),
            destination_feats=dest_feats,
            destination_idx=dest_idx,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            was_parking=False
        )
        
        return HierarchicalStepResult(
            next_state=next_state,
            reward=reward,
            done=done,
            info=info,
            transition=transition,
            was_valid_move=success
        )
    
    def _determine_move_type(self, source: SourceType, dest: DestinationType) -> MoveType:
        """Map source/destination to MoveType."""
        mapping = {
            (SourceType.YARD, DestinationType.YARD): MoveType.YARD_TO_YARD,
            (SourceType.YARD, DestinationType.TRAIN): MoveType.YARD_TO_TRAIN,
            (SourceType.YARD, DestinationType.TRUCK): MoveType.YARD_TO_TRUCK,
            (SourceType.TRAIN, DestinationType.YARD): MoveType.TRAIN_TO_YARD,
            (SourceType.TRAIN, DestinationType.TRUCK): MoveType.TRAIN_TO_TRUCK,
            (SourceType.TRUCK, DestinationType.YARD): MoveType.TRUCK_TO_YARD,
            (SourceType.TRUCK, DestinationType.TRAIN): MoveType.TRUCK_TO_TRAIN,
        }
        return mapping.get((source, dest), MoveType.YARD_TO_YARD)
    
    def _build_move_args(self, cont: MoveableContainer, dest: Destination) -> Dict[str, Any]:
        """Build move args dictionary."""
        args = {"container_id": cont.container_id}
        
        if cont.source_type == SourceType.TRAIN:
            args["train_id"] = cont.source_id
        elif cont.source_type == SourceType.TRUCK:
            args["truck_id"] = cont.source_id
        
        if dest.dest_type == DestinationType.YARD:
            args["placement"] = PlacementResult(
                row=dest.row,
                bay=dest.bay,
                tier=dest.tier,
                start_split=dest.start_split
            )
        elif dest.dest_type == DestinationType.TRAIN:
            args["train_id"] = dest.dest_id
        elif dest.dest_type == DestinationType.TRUCK:
            args["truck_id"] = dest.dest_id
        
        return args
    
    def _create_move(self, move_type: MoveType, args: Dict) -> Any:
        """Create Move object compatible with TLM."""
        from simulation.operations.terminal_manager import Move
        return Move(type=move_type, args=args)
    
    def _advance_time(self):
        """Advance simulation time by step_minutes."""
        self.env.current_time += timedelta(minutes=self.env.step_minutes)
    
    def _check_day_end(self) -> bool:
        """Check if day has ended."""
        if self.env.day_plan is None:
            return True
        day_end = self.env.day_plan.date.replace(hour=23, minute=59, second=0)
        return self.env.current_time >= day_end
    
    def step_all_cranes(
        self,
        agent: HierarchicalDQNAgent
    ) -> Tuple[NDArray[np.float32], float, bool, Dict[str, Any]]:
        """
        Execute hierarchical steps for all available cranes.
        
        OPTIMIZED: Encodes state once, passes between crane steps,
        only re-encodes when moves actually change state.
        """
        total_reward = 0.0
        all_info: Dict[str, Any] = {
            "crane_results": [],
            "executed": [],
            "transitions": []
        }
        done = False
        
        # Handle arrivals/departures first
        self._handle_arrivals_departures()
        
        now = self.env.current_time
        idle_cranes = [
            c for c in self.env.cranes
            if c.busy_until is None or c.busy_until <= now
        ]
        
        # Encode state ONCE at start
        state = self._encode_state()
        
        if not idle_cranes:
            # All cranes busy - advance to next available
            next_time = min(
                c.busy_until for c in self.env.cranes
                if c.busy_until is not None
            )
            self.env.current_time = next_time
            # Time changed but yard didn't - still need fresh encode for time features
            state = self._encode_state()
            done = self._check_day_end()
            return state, 0.0, done, all_info
        
        # Process each idle crane, passing state between them
        for crane in idle_cranes:
            result = self.step_hierarchical(agent, crane_id=crane.id, state_np=state)
            
            # Update state for next crane (result contains updated state if move succeeded)
            state = result.next_state
            
            total_reward += result.reward
            all_info["crane_results"].append({
                "crane_id": crane.id,
                "reward": result.reward,
                "was_valid": result.was_valid_move
            })
            all_info["executed"].extend(result.info.get("executed", []))
            
            if result.transition is not None:
                all_info["transitions"].append(result.transition)
                agent.remember(result.transition)
            
            if result.done:
                done = True
                break
        
        # Final state is already computed - no need to re-encode!
        return state, total_reward, done, all_info
    
    def _handle_arrivals_departures(self):
        """Handle train/truck arrivals and departures."""
        # Delegate to env's internal handling if available
        if hasattr(self.env, '_admit_arrivals_and_departures'):
            self.env._admit_arrivals_and_departures()
        
        # Update train heat after potential arrivals
        self._update_train_heat()